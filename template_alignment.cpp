#include "template_alignment.h"

float cam_position_x = 0.410;
float cam_position_y = -0.2565;
float cam_position_z = 0.321;
float cam_orientation_x = -0.660;
float cam_orientation_y = -0.652;
float cam_orientation_z = 0.245;
float cam_orientation_w = 0.282;

float hole_position_x = 0.619;
float hole_position_y = -0.254;
float hole_position_z = 0.1996;
float hole_orientation_x = -0.0;
float hole_orientation_y = -0.0;
float hole_orientation_z = -0.0;
float hole_orientation_w = 1.0;

float tar_cam_position_x = 0.410;
float tar_cam_position_y = -0.2565;
float tar_cam_position_z = 0.321;
float tar_cam_orientation_x = -0.660;
float tar_cam_orientation_y = -0.652;
float tar_cam_orientation_z = 0.245;
float tar_cam_orientation_w = 0.282;

void
TemplateAlign(char* target_pcd_path)
{
  // load template file
  std::vector<FeatureCloud> object_templates;
  std::ifstream input_stream(py2cpp_.template_list_path);
  object_templates.resize(0);
  std::string pcd_filename;
  while(input_stream.good())
  {
    std::getline(input_stream,pcd_filename);
    if(pcd_filename.empty()||pcd_filename.at(0) == '#')
      continue;

    FeatureCloud template_cloud;
    template_cloud.loadInputCloud(pcd_filename);
    object_templates.push_back(template_cloud);
  }

  input_stream.close();

  // load the traget cloud PCD file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::io::loadPCDFile(target_pcd_path,*cloud);

  // Preprocess the cloud by...
  // ...removing distant points
  const float depth_limit = 1.0;
  pcl::PassThrough<pcl::PointXYZ> pass;
  pass.setInputCloud (cloud);
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0, depth_limit);
  pass.filter (*cloud);

  // ... and downsampling the point cloud
  const float voxel_grid_size = 0.003f;
  pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
  vox_grid.setInputCloud (cloud);
  vox_grid.setLeafSize (voxel_grid_size, voxel_grid_size, voxel_grid_size);
  vox_grid.filter (*cloud);

  // Assign to the target FeatureCloud
  FeatureCloud target_cloud;
  target_cloud.setInputCloud(cloud);

  // Set the TemplateAlignment inputs
  TemplateAlignment template_align;
  for (size_t i = 0;i<object_templates.size();i++)
  {
    template_align.addTemplateCloud(object_templates[i]);
  }
  template_align.setTargetCloud(target_cloud);

  // Find the best template alignment
  TemplateAlignment::Result best_alignment;
  int best_index = template_align.findBestAlignment(best_alignment);
  const FeatureCloud &best_template = object_templates[best_index];

  // Print the alignment fitness score (values less than 0.00002 are good)
  printf ("Best fitness score: %f\n", best_alignment.fitness_score);

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaxCorrespondenceDistance(100);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(best_template.getPointCloud());
  icp.setInputTarget(target_cloud.getPointCloud());
  pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZ>());
  icp.align(*unused_result, best_alignment.final_transformation);

  // Print the rotation matrix and translation vector
  Eigen::Matrix3f rotation = icp.getFinalTransformation().block<3,3>(0, 0);
  Eigen::Vector3f translation = icp.getFinalTransformation().block<3,1>(0, 3);

  printf ("\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

  Eigen::Quaternionf trans_camera2world(cam_orientation_w, cam_orientation_x, cam_orientation_y, cam_orientation_z);
  Eigen::Vector4f pos_vec(cam_position_x, cam_position_y, cam_position_z, 1.0);
  Eigen::Matrix4f trans_camera2world_mat = Eigen::Matrix4f::Zero();
  trans_camera2world_mat.block<3, 3>(0, 0) = trans_camera2world.matrix();
  trans_camera2world_mat.block<4, 1>(0, 3) = pos_vec;

  Eigen::Quaternionf trans_hole2world(hole_orientation_w, hole_orientation_x, hole_orientation_y, hole_orientation_z);
  Eigen::Vector4f hole_pos_vec(hole_position_x, hole_position_y, hole_position_z, 1.0);
  Eigen::Matrix4f trans_hole2world_mat = Eigen::Matrix4f::Zero();
  trans_hole2world_mat.block<3, 3>(0, 0) = trans_hole2world.matrix();
  trans_hole2world_mat.block<4, 1>(0, 3) = hole_pos_vec;

  Eigen::Matrix4f hole2cam = trans_camera2world_mat.inverse() * trans_hole2world_mat;

  Eigen::Matrix4f hole2cam_target = icp.getFinalTransformation() * hole2cam;

  //////////////////////////
  StandardTrans standardtrans_;
  Eigen::Matrix4f trans_camera2target = icp.getFinalTransformation();
  Eigen::Matrix4f trans_hole2target = trans_camera2target * standardtrans_.trans_hole2camera;
  Eigen::Matrix4f trans_target2base = Eigen::Matrix4f::Identity(); // get from robot
  Eigen::Matrix4f trans_hole2base = trans_target2base * trans_hole2target;
  ///////////////////////////

  rotation = trans_hole2base.block<3,3>(0, 0);
  translation = trans_hole2base.block<3,1>(0, 3);

  printf ("\n");
  printf ("hole to base from estimation: \n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

  Eigen::Quaternionf tar_trans_camera2world(tar_cam_orientation_w, tar_cam_orientation_x, tar_cam_orientation_y, tar_cam_orientation_z);
  Eigen::Vector4f tar_pos_vec(tar_cam_position_x, tar_cam_position_y, tar_cam_position_z, 1.0);
  Eigen::Matrix4f tar_trans_camera2world_mat = Eigen::Matrix4f::Zero();
  tar_trans_camera2world_mat.block<3, 3>(0, 0) = tar_trans_camera2world.matrix();
  tar_trans_camera2world_mat.block<4, 1>(0, 3) = tar_pos_vec;

  Eigen::Matrix4f tar_hole2cam = tar_trans_camera2world_mat.inverse() * trans_hole2world_mat;

  rotation = tar_hole2cam.block<3,3>(0, 0);
  translation = tar_hole2cam.block<3,1>(0, 3);

  printf ("\n");
  printf ("hole to camera from optitrack:\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

  var2py_.best_fitness_score = best_alignment.fitness_score;
  var2py_.rotation_00 = rotation (0,0);
  var2py_.rotation_01 = rotation (0,1);
  var2py_.rotation_02 = rotation (0,2);
  var2py_.rotation_10 = rotation (1,0);
  var2py_.rotation_11 = rotation (1,1);
  var2py_.rotation_12 = rotation (1,2);
  var2py_.rotation_20 = rotation (2,0);
  var2py_.rotation_21 = rotation (2,1);
  var2py_.rotation_22 = rotation (2,2);
  var2py_.translation_x = translation (0);
  var2py_.translation_y = translation (1);
  var2py_.translation_z = translation (2);

  // Save the aligned template for visualization
  pcl::PointCloud<pcl::PointXYZ>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZ>);
  pcl::transformPointCloud (*best_template.getPointCloud (), *transformed_cloud, icp.getFinalTransformation());
  pcl::io::savePCDFileBinary ("output.pcd", *transformed_cloud);
  
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis(transformed_cloud, cloud);
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    boost::this_thread::sleep (boost::posix_time::microseconds (100000));
  }
}