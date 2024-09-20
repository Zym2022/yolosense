#include "template_alignment.h"

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

  pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
  ndt.setTransformationEpsilon(0.01);
  ndt.setResolution(1.0);

  ndt.setInputSource(best_template.getPointCloud());
  ndt.setInputTarget(target_cloud.getPointCloud());
  pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result_0(new pcl::PointCloud<pcl::PointXYZ>());

  ndt.align(*unused_result_0, best_alignment.final_transformation);

  pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
  icp.setMaxCorrespondenceDistance(100);
  icp.setMaximumIterations(100);
  icp.setTransformationEpsilon(1e-6);
  icp.setEuclideanFitnessEpsilon(1e-6);
  icp.setRANSACIterations(0);

  icp.setInputSource(best_template.getPointCloud());
  icp.setInputTarget(target_cloud.getPointCloud());
  pcl::PointCloud<pcl::PointXYZ>::Ptr unused_result(new pcl::PointCloud<pcl::PointXYZ>());
  icp.align(*unused_result, ndt.getFinalTransformation());

  // Print the rotation matrix and translation vector
  Eigen::Matrix3f rotation = icp.getFinalTransformation().block<3,3>(0, 0);
  Eigen::Vector3f translation = icp.getFinalTransformation().block<3,1>(0, 3);

  printf ("\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

  //////////////////////////
  // StandardTrans standardtrans_;
  // Eigen::Matrix4f trans_camera2target = icp.getFinalTransformation();
  // Eigen::Matrix4f trans_hole2target = trans_camera2target * standardtrans_.trans_hole2camera;
  // Eigen::Matrix4f trans_end2base = Eigen::Matrix4f::Identity(); // get from robot
  // trans_end2base << -0.4999368190765381, -1.5891189832473174e-05, 0.8660618662834167, 0.7147712707519531,
  //                   -1.0617844964144751e-05, 1.0, 1.221960974362446e-05, -0.1500995010137558,
  //                   -0.8660618662834167, -3.086678134422982e-06, -0.4999368190765381, 0.16625721752643585,
  //                   0.0, 0.0, 0.0, 1.0;    
  // Eigen::Matrix4f trans_target2base = trans_end2base * standardtrans_.trans_camera2end;
  // Eigen::Matrix4f trans_hole2base = trans_target2base * trans_hole2target;
  ///////////////////////////

  ////////////////
  StandardTrans standardtrans_;
  Eigen::Matrix4f trans_end2base;
  trans_end2base << py2cpp_.end00, py2cpp_.end01, py2cpp_.end02, py2cpp_.end03, 
                    py2cpp_.end10, py2cpp_.end11, py2cpp_.end12, py2cpp_.end13,
                    py2cpp_.end20, py2cpp_.end21, py2cpp_.end22, py2cpp_.end23,
                    py2cpp_.end30, py2cpp_.end31, py2cpp_.end32, py2cpp_.end33;
  
  // trans_end2base << -0.4999368190765381, -1.5891189832473174e-05, 0.8660618662834167, 0.7147712707519531,
  //                   -1.0617844964144751e-05, 1.0, 1.221960974362446e-05, -0.1500995010137558,
  //                   -0.8660618662834167, -3.086678134422982e-06, -0.4999368190765381, 0.16625721752643585,
  //                   0.0, 0.0, 0.0, 1.0;    

  std::cout << trans_end2base << std::endl;
  Eigen::Matrix4f trans_target2camera = icp.getFinalTransformation() * standardtrans_.trans_hole2camera;
  Eigen::Matrix4f trans_target2base = trans_end2base * standardtrans_.trans_camera2end * trans_target2camera;
  ////////////////

  rotation = trans_target2base.block<3,3>(0, 0);
  translation = trans_target2base.block<3,1>(0, 3);

  printf ("\n");
  printf ("hole to base from estimation: \n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));


  Eigen::AngleAxisf rotationV(M_PI*3/4, Eigen::Vector3f(0, 1, 0));
  Eigen::Matrix4f rotationMatrix = Eigen::Matrix4f::Identity();
  rotationMatrix.block<3, 3>(0, 0) = rotationV.toRotationMatrix();
  Eigen::Matrix4f Result = trans_target2base * rotationMatrix;

  rotation = Result.block<3,3>(0, 0);
  translation = Result.block<3,1>(0, 3);

  printf ("\n");
  printf ("transformed hole to base from estimation: \n");
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

  float hole_position_x_val = 0.92908;
  float hole_position_y_val = -0.31034;
  float hole_position_z_val = 0.11495;
  float hole_orientation_x_val = 0.0;
  float hole_orientation_y_val = 0.0;
  float hole_orientation_z_val = 0.0;
  float hole_orientation_w_val = 1.0;

  Eigen::Matrix4f trans_hole2world_val = Eigen::Matrix4f::Identity();
  Eigen::Quaternionf quat_hole2world_val(hole_orientation_w_val, hole_orientation_x_val, hole_orientation_y_val, hole_orientation_z_val);
  Eigen::Vector4f hole_pos_vec_val(hole_position_x_val, hole_position_y_val, hole_position_z_val, 1.0);
  trans_hole2world_val.block<3, 3>(0, 0) = quat_hole2world_val.matrix();
  trans_hole2world_val.block<4, 1>(0, 3) = hole_pos_vec_val;
  Eigen::Matrix4f trans_hole2base_val = standardtrans_.trans_base2world.inverse() * trans_hole2world_val;
  trans_hole2base_val = trans_hole2base_val * rotationMatrix;

  rotation = trans_hole2base_val.block<3,3>(0, 0);
  translation = trans_hole2base_val.block<3,1>(0, 3);

  printf ("\n");
  printf ("tranformed hole to base from optitrack:\n");
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (0,0), rotation (0,1), rotation (0,2));
  printf ("R = | %6.3f %6.3f %6.3f | \n", rotation (1,0), rotation (1,1), rotation (1,2));
  printf ("    | %6.3f %6.3f %6.3f | \n", rotation (2,0), rotation (2,1), rotation (2,2));
  printf ("\n");
  printf ("t = < %0.3f, %0.3f, %0.3f >\n", translation (0), translation (1), translation (2));

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