#include <Eigen/Geometry>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>
#include <mutex>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/common/transforms.h>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <queue>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>

#include "include/STDesc.h"
#include "ros/init.h"

typedef pcl::PointXYZI PointType;
typedef pcl::PointCloud<PointType> PointCloud;

std::mutex laser_mtx;
std::mutex odom_mtx;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
  std::unique_lock<std::mutex> lock(laser_mtx);

  laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
  std::unique_lock<std::mutex> lock(odom_mtx);

  odom_buffer.push(msg);
}

bool syncPackages(PointCloud::Ptr &cloud, Eigen::Affine3d &pose) {
  if (laser_buffer.empty() || odom_buffer.empty())
    return false;

  auto laser_msg = laser_buffer.front();
  double laser_timestamp = laser_msg->header.stamp.toSec();

  auto odom_msg = odom_buffer.front();
  double odom_timestamp = odom_msg->header.stamp.toSec();

  // check if timestamps are matched
  if (abs(odom_timestamp - laser_timestamp) < 1e-3) {
    pcl::fromROSMsg(*laser_msg, *cloud);

    Eigen::Quaterniond r(
        odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
    Eigen::Vector3d t(odom_msg->pose.pose.position.x,
                      odom_msg->pose.pose.position.y,
                      odom_msg->pose.pose.position.z);

    pose = Eigen::Affine3d::Identity();
    pose.translate(t);
    pose.rotate(r);

    std::unique_lock<std::mutex> l_lock(laser_mtx);
    std::unique_lock<std::mutex> o_lock(odom_mtx);

    laser_buffer.pop();
    odom_buffer.pop();

  } else if (odom_timestamp < laser_timestamp) {
    ROS_WARN("Current odometry is earlier than laser scan, discard one "
             "odometry data.");
    std::unique_lock<std::mutex> o_lock(odom_mtx);
    odom_buffer.pop();
    return false;
  } else {
    ROS_WARN(
        "Current laser scan is earlier than odometry, discard one laser scan.");
    std::unique_lock<std::mutex> l_lock(laser_mtx);
    laser_buffer.pop();
    return false;
  }

  return true;
}

void update_poses(const gtsam::Values &estimates,
                  std::vector<Eigen::Affine3d> &poses) {
  assert(estimates.size() == poses.size()); // 注意 这个是相等的

  poses.clear();

  for (int i = 0; i < estimates.size(); ++i) {
    auto est = estimates.at<gtsam::Pose3>(i);
    Eigen::Affine3d est_affine3d(est.matrix());
    poses.push_back(est_affine3d);
  }
}

void visualizeLoopClosure(
    const ros::Publisher &publisher,
    const std::vector<std::pair<int, int>> &loop_container,
    const std::vector<Eigen::Affine3d> &key_pose_vec) {
  if (loop_container.empty())
    return;

  visualization_msgs::MarkerArray markerArray;
  // 闭环顶点
  visualization_msgs::Marker markerNode;
  markerNode.header.frame_id = "camera_init"; // camera_init
  // markerNode.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
  markerNode.action = visualization_msgs::Marker::ADD;
  markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
  markerNode.ns = "loop_nodes";
  markerNode.id = 0;
  markerNode.pose.orientation.w = 1;
  markerNode.scale.x = 0.3;
  markerNode.scale.y = 0.3;
  markerNode.scale.z = 0.3;
  markerNode.color.r = 0;
  markerNode.color.g = 0.8;
  markerNode.color.b = 1;
  markerNode.color.a = 1;
  // 闭环边
  visualization_msgs::Marker markerEdge;
  markerEdge.header.frame_id = "camera_init";
  // markerEdge.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
  markerEdge.action = visualization_msgs::Marker::ADD;
  markerEdge.type = visualization_msgs::Marker::LINE_LIST;
  markerEdge.ns = "loop_edges";
  markerEdge.id = 1;
  markerEdge.pose.orientation.w = 1;
  markerEdge.scale.x = 0.1;
  markerEdge.color.r = 0.9;
  markerEdge.color.g = 0.9;
  markerEdge.color.b = 0;
  markerEdge.color.a = 1;

  // 遍历闭环
  for (auto it = loop_container.begin(); it != loop_container.end(); ++it) {
    int key_cur = it->first;
    int key_pre = it->second;
    geometry_msgs::Point p;
    p.x = key_pose_vec[key_cur].translation().x();
    p.y = key_pose_vec[key_cur].translation().y();
    p.z = key_pose_vec[key_cur].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
    p.x = key_pose_vec[key_pre].translation().x();
    p.y = key_pose_vec[key_pre].translation().y();
    p.z = key_pose_vec[key_pre].translation().z();
    markerNode.points.push_back(p);
    markerEdge.points.push_back(p);
  }

  markerArray.markers.push_back(markerNode);
  markerArray.markers.push_back(markerEdge);
  publisher.publish(markerArray);
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "online_demo");
  ros::NodeHandle nh;

  ConfigSetting config_setting;
  read_parameters(nh, config_setting);

  // ros::Publisher pubOdomAftMapped =
  // nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
  ros::Publisher pubCurrentCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
  ros::Publisher pubCurrentCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
  ros::Publisher pubMatchedCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
  ros::Publisher pubMatchedCorner =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
  ros::Publisher pubSTD =
      nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

  ros::Publisher pubOriginCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);

  ros::Publisher pubCorrectCloud =
      nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
  ros::Publisher pubCorrectPath =
      nh.advertise<nav_msgs::Path>("/correct_path", 100000);

  ros::Publisher pubOdomOrigin =
      nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);
  ros::Publisher pubLoopConstraintEdge =
      nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints",
                                                    10);

  ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(
      "/cloud_registered_body", 100, laserCloudHandler);
  ros::Subscriber subOdom =
      nh.subscribe<nav_msgs::Odometry>("/Odometry", 100, OdomHandler);

  STDescManager *std_manager = new STDescManager(config_setting);

  gtsam::Values initial; // 创建一个空容器(本质是key->value字典)
  gtsam::NonlinearFactorGraph graph; // 创建一个空因子图(本质是约束集合，每个约束对应一个因子Factor)

  // https://github.com/TixiaoShan/LIO-SAM/blob/6665aa0a4fcb5a9bb3af7d3923ae4a035b489d47/src/mapOptmization.cpp#L1385
  // 六维对角高斯噪声模型，用来描述里程计约束的不确定性 (roll, pitch, yaw, x, y, z)
  // 等价于: 
  // auto odometryNoise =
  //   gtsam::noiseModel::Diagonal::Sigmas( // Sigmas是使用标准差而不是方差来定义噪声模型
  //       (gtsam::Vector(6) <<
  //           0.001, 0.001, 0.001,
  //           0.01,  0.01,  0.01
  //       ).finished());
  gtsam::noiseModel::Diagonal::shared_ptr odometryNoise =
      gtsam::noiseModel::Diagonal::Variances( // Variances是使用方差而不是标准差来定义噪声模型
          (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());

  gtsam::noiseModel::Diagonal::shared_ptr priorNoise =
      gtsam::noiseModel::Diagonal::Variances(
          (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8)
              .finished()); // rad*rad, meter*meter

  double loopNoiseScore = 1e-1;
  gtsam::Vector robustNoiseVector6(
      6); // gtsam::Pose3 factor has 6 elements (6D)
  robustNoiseVector6 << loopNoiseScore, loopNoiseScore, loopNoiseScore,
      loopNoiseScore, loopNoiseScore, loopNoiseScore;
  gtsam::noiseModel::Base::shared_ptr robustLoopNoise =
      gtsam::noiseModel::Robust::Create(
          gtsam::noiseModel::mEstimator::Cauchy::Create(1),
          gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));
  // 设置重线性化阈值和跳过次数
  // 这里表示每次 update() 都检查重线性化条件，变量相对线性化点的改变量超过阈值时，触发相应处理。
  // 若没有达到条件，第二次更新可能几乎不改变结果
  gtsam::ISAM2Params parameters;
  parameters.relinearizeThreshold = 0.01;
  parameters.relinearizeSkip = 1;
  gtsam::ISAM2 isam(parameters); // 创建一个增量平滑求解器

  size_t cloudInd = 0;    // 每个成功同步的普通单帧点云索引
  size_t keyCloudInd = 0; // 多帧累积的关键帧点云索引

  std::vector<PointCloud::Ptr> cloud_vec; // 存储单帧原始点云
  std::vector<Eigen::Affine3d> pose_vec;  // 存储优化后位姿
  std::vector<Eigen::Affine3d> origin_pose_vec; // 存储优化前位姿
  std::vector<Eigen::Affine3d> key_pose_vec;    // 存储所有关键帧位姿
  std::vector<std::pair<int, int>> loop_container; // 存储所有回环检测结果的索引对

  PointCloud::Ptr key_cloud(new PointCloud); // 存储当前关键帧点云(多帧累积)

  bool has_loop_flag = false;
  gtsam::Values curr_estimate;

  Eigen::Affine3d last_pose;
  last_pose.setIdentity();
  while (ros::ok()) {
    ros::spinOnce();

    PointCloud::Ptr current_cloud_body(new PointCloud);
    PointCloud::Ptr current_cloud_world(new PointCloud);
    Eigen::Affine3d pose;
    // 获取同步后的点云和里程计位姿信息
    if (syncPackages(current_cloud_body, pose)) {
      auto origin_estimate_affine3d = pose;
      pcl::transformPointCloud(*current_cloud_body, *current_cloud_world, pose);
      down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
      // down sample body cloud
      down_sampling_voxel(*current_cloud_body, 0.5);
      cloud_vec.push_back(current_cloud_body);
      pose_vec.push_back(pose);
      origin_pose_vec.push_back(pose);
      PointCloud origin_cloud;
      pcl::transformPointCloud(*current_cloud_body, origin_cloud,
                               origin_estimate_affine3d);
      sensor_msgs::PointCloud2 pub_cloud;
      pcl::toROSMsg(origin_cloud, pub_cloud);
      pub_cloud.header.frame_id = "camera_init";
      pubOriginCloud.publish(pub_cloud); // 发布当前帧点云(world坐标系下)

      Eigen::Quaterniond _r(origin_estimate_affine3d.rotation());
      nav_msgs::Odometry odom;
      odom.header.frame_id = "camera_init";
      odom.pose.pose.position.x = origin_estimate_affine3d.translation().x();
      odom.pose.pose.position.y = origin_estimate_affine3d.translation().y();
      odom.pose.pose.position.z = origin_estimate_affine3d.translation().z();
      odom.pose.pose.orientation.w = _r.w();
      odom.pose.pose.orientation.x = _r.x();
      odom.pose.pose.orientation.y = _r.y();
      odom.pose.pose.orientation.z = _r.z();
      pubOdomOrigin.publish(odom); // 发布优化前位姿
      // world坐标系下点云先滤波，再累积
      *key_cloud += *current_cloud_world; // 点云累积，用于后面提取描述子进行回环检测
      initial.insert(cloudInd, gtsam::Pose3(pose.matrix()));

      if (!cloudInd) { // 添加首帧先验因子到因子图
        graph.add(gtsam::PriorFactor<gtsam::Pose3>(
            0, gtsam::Pose3(pose.matrix()), odometryNoise));
      } else { // 添加里程计约束因子到因子图
        auto prev_pose = gtsam::Pose3(origin_pose_vec[cloudInd - 1].matrix());
        auto curr_pose = gtsam::Pose3(pose.matrix());
        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
            cloudInd - 1, cloudInd, prev_pose.between(curr_pose),
            odometryNoise));
      }

      // check if keyframe
      // sub_frame_num_参数在这里有两个作用: 
      // 1. 控制点云累积和描述子提取的间隔;
      // 2. 控制一次闭环添加多少条 GTSAM 约束
      if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0) {
        ROS_INFO("key frame idx: [%d], key cloud size: [%d]", (int)keyCloudInd,
                 (int)key_cloud->size());
        // step1. Descriptor Extraction
        std::vector<STDesc> stds_vec;
        std_manager->GenerateSTDescs(key_cloud, stds_vec);

        // step2. Searching Loop
        // search_result.first   // 回环找到的历史关键点云编号，对应 keyCloudInd 这一层
        // search_result.second  // 几何验证得分，越大越好
        std::pair<int, double> search_result(-1, 0); // 找到哪份历史点云
        std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
        loop_transform.first << 0, 0, 0; // 平移
        loop_transform.second = Eigen::Matrix3d::Identity(); // 旋转
        // loop_std_pair[i].first   // 当前关键点云中的描述子
        // loop_std_pair[i].second  // 匹配的历史关键点云中的描述子
        std::vector<std::pair<STDesc, STDesc>> loop_std_pair; // 匹配成功的描述子对
        // 等积累了足够多的累积关键帧后，才开始进行回环检测
        // 所以没有积累足够多的累积关键帧之前，都是进行描述子提取、保存描述子，积累历史数据库的步骤
        if (keyCloudInd > config_setting.skip_near_num_) {
          std_manager->SearchLoop(stds_vec, search_result, loop_transform,
                                  loop_std_pair);
        }

        // step3. Add descriptors to the database
        std_manager->AddSTDescs(stds_vec);

        // publish
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(*key_cloud, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCurrentCloud.publish(pub_cloud); // 发布当前关键帧累积点云
        pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCurrentCorner.publish(pub_cloud); // 发布当前关键帧累积角点云

        std_manager->key_cloud_vec_.push_back(key_cloud->makeShared());
        // 检测到回环(没有合格闭环时，返回 {-1, 0})
        if (search_result.first > 0) {
          std::cout << "[Loop Detection] triggle loop: " << keyCloudInd << "--"
                    << search_result.first << ", score:" << search_result.second
                    << std::endl;

          has_loop_flag = true;
          int match_frame = search_result.first; // 对应回环找到的历史关键点云编号 keyCloudInd
          // obtain optimal transform
          std_manager->PlaneGeomrtricIcp(
              std_manager->plane_cloud_vec_.back(),
              std_manager->plane_cloud_vec_[match_frame], loop_transform);

          // std::cout << "delta transform:" << std::endl;
          // std::cout << "translation: " << loop_transform.first.transpose() <<
          // std::endl;

          // auto euler = loop_transform.second.eulerAngles(2, 1, 0) * 57.3;
          // std::cout << "rotation(ypr): " << euler[0] << ' ' << euler[1] << '
          // ' << euler[2]
          //           << std::endl;

          /*
            add connection between loop frame.
            e.g. if src_key_frame_id 5 with sub frames 51~60 triggle loop with
                  tar_key_frame_id 1 with sub frames 11~20, add connection
            between each sub frame, 51-11, 52-12,...,60-20.

          */
          // 因为回环检测是基于累积的关键帧的(每个累积关键帧由 sub_frame_num 个普通帧组成)，
          // 所以在添加回环约束时，需要对每个普通子帧进行相应的处理
          int sub_frame_num = config_setting.sub_frame_num_;
          // 其实就是 [cloudInd-9, cloudInd]
          for (size_t j = 1; j <= sub_frame_num; j++) {
            int src_frame = cloudInd + j - sub_frame_num; // 当前普通帧

            auto delta_T = Eigen::Affine3d::Identity(); // T_tar_src
            delta_T.translate(loop_transform.first);
            delta_T.rotate(loop_transform.second);
            // 经过回环变换后补偿的当前普通帧位姿
            Eigen::Affine3d src_pose_refined = delta_T * pose_vec[src_frame];
            // match_frame 是基于 keyCloudInd 的，下面就是这个累积关键帧对应到普通帧的换算关系
            int tar_frame = match_frame * sub_frame_num + j; // 匹配到的历史普通帧
            // old
            // Eigen::Affine3d tar_pose = pose_vec[tar_frame];
            Eigen::Affine3d tar_pose = origin_pose_vec[tar_frame];

            loop_container.push_back({tar_frame, src_frame});
            // 添加回环约束因子到因子图
            graph.add(gtsam::BetweenFactor<gtsam::Pose3>(
                tar_frame, src_frame,
                gtsam::Pose3(tar_pose.matrix())
                    .between(gtsam::Pose3(src_pose_refined.matrix())),
                robustLoopNoise));
          }

          pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first],
                        pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCloud.publish(pub_cloud); // 发布回环匹配到的累积关键帧点云

          pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first],
                        pub_cloud);
          pub_cloud.header.frame_id = "camera_init";
          pubMatchedCorner.publish(pub_cloud); // 发布回环匹配到的累积关键帧角点云
          publish_std_pairs(loop_std_pair, pubSTD);
        }

        key_cloud->clear();
        ++keyCloudInd;
      }
      // graph 包含本轮新增的里程计约束，以及可能检测到的闭环约束；initial 提供新节点的初始位姿
      // iSAM2 将它们加入内部保存的历史图中，更新求解结果。闭环约束可能使多个历史节点的位姿一起改变
      // sam.update(graph, initial); 这一行本身已经执行求解更新
      isam.update(graph, initial); // 加入本轮新数据，并更新求解结果
      // 不传参数时，使用默认的空因子图和空变量集合，等价于：
      // isam.update(gtsam::NonlinearFactorGraph(), gtsam::Values());
      // 这次没有添加新数据，但 iSAM2 内部仍保存着之前提交的全部约束，因此可以继续更新已有问题
      isam.update();               // 基于已保存的图，再更新一次
      // 连续调用的原因是：位姿优化是非线性问题，一次更新后，可能需要在调整后的位姿附近重新计算线性近似，
      // 再继续修正。 这个过程叫“重线性化”

      // 如果检测到闭环，则进行多次重线性化更新，确保位姿优化收敛，也是为了让闭环带来的较大修正继续得到处理。
      if (has_loop_flag) {
        isam.update();
        isam.update();
        isam.update();
        isam.update();
        isam.update();
      }
      // 普通帧调用 2 次更新，闭环帧调用 7 次更新；这些调用都按阈值决定实际需要做多少工作

      // 清空本轮新增的因子图和初始值，为下一轮更新做准备
      // graph 中保存本轮新增的里程计约束、闭环约束。执行后 graph.size() == 0
      // 下一帧再往里面添加新约束。
      // 之前通过 isam.update() 提交的因子仍保存在求解器中!!!
      graph.resize(0); // 清空本轮新增的因子图
      initial.clear(); // 清空本轮新增变量的初始值
      // 比如本轮添加的是：initial.insert(5, pose5);
      // 那么提交并清空后，下一轮就可以只放第 6 帧， initial.insert(6, pose6);
      // 第 5 帧的状态已经由 isam 管理
      /*
      整个流程就是：
      1. 准备本轮新增约束和新节点初值;
      2. 提交给持续保存历史状态的求解器;
      3. 清空临时容器，下一轮重新装入新数据.   
      这样可以避免下一轮重复提交旧因子和已有变量.
      重复提交因子会重复计入约束；initial 则要求只包含尚未加入 isam 的新变量。
      */

      // 取得目前位姿图中，所有节点的位姿估计结果
      curr_estimate = isam.calculateEstimate();
      update_poses(curr_estimate, pose_vec);

      auto latest_estimate_affine3d = pose_vec.back();

      if (has_loop_flag) {
        // publish correct cloud map
        PointCloud full_map;
        for (int i = 0; i < pose_vec.size(); ++i) {
          PointCloud correct_cloud;
          pcl::transformPointCloud(*cloud_vec[i], correct_cloud, pose_vec[i]);
          full_map += correct_cloud;
        }
        sensor_msgs::PointCloud2 pub_cloud;
        pcl::toROSMsg(full_map, pub_cloud);
        pub_cloud.header.frame_id = "camera_init";
        pubCorrectCloud.publish(pub_cloud);

        // publish corerct path
        nav_msgs::Path correct_path;
        for (int i = 0; i < pose_vec.size(); i += 1) {

          geometry_msgs::PoseStamped msg_pose;
          msg_pose.pose.position.x = pose_vec[i].translation()[0];
          msg_pose.pose.position.y = pose_vec[i].translation()[1];
          msg_pose.pose.position.z = pose_vec[i].translation()[2];
          Eigen::Quaterniond pose_q(pose_vec[i].rotation());
          msg_pose.header.frame_id = "camera_init";
          msg_pose.pose.orientation.x = pose_q.x();
          msg_pose.pose.orientation.y = pose_q.y();
          msg_pose.pose.orientation.z = pose_q.z();
          msg_pose.pose.orientation.w = pose_q.w();
          correct_path.poses.push_back(msg_pose);
        }
        correct_path.header.stamp = ros::Time::now();
        correct_path.header.frame_id = "camera_init";
        pubCorrectPath.publish(correct_path);
      }
      //   PointCloud correct_cloud;
      //   pcl::transformPointCloud(*current_cloud_body, correct_cloud,
      //                            latest_estimate_affine3d);
      //   sensor_msgs::PointCloud2 pub_cloud;
      //   pcl::toROSMsg(correct_cloud, pub_cloud);
      //   pub_cloud.header.frame_id = "camera_init";
      //   pubCorrectCloud.publish(pub_cloud);

      visualizeLoopClosure(pubLoopConstraintEdge, loop_container, pose_vec);

      has_loop_flag = false;
      ++cloudInd;
    }
  }

  // You can save full map with refined pose
  // assert(cloud_vec.size() == pose_vec.size());
  // PointCloud full_map;
  // for (int i = 0; i < pose_vec.size(); ++i) {
  //     PointCloud correct_cloud;
  //     pcl::transformPointCloud(*cloud_vec[i], correct_cloud, pose_vec[i]);
  //     full_map += correct_cloud;
  // }
  // down_sampling_voxel(full_map, 0.05);

  // std::cout << "saving map..." << std::endl;
  // pcl::io::savePCDFileBinary("/home/dustier/data/map.pcd", full_map);

  return 0;
}
