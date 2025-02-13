#include <limits>
#include <fstream>
#include <string>
#include <vector>
#include <Eigen/Core> 
#include <boost/thread/thread.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>

#include <tbb/parallel_for.h>


// put the variables needed by python into this struct
struct Var2Py {
  float best_fitness_score;
  float rotation_00;
  float rotation_01;
  float rotation_02;
  float rotation_10;
  float rotation_11;
  float rotation_12;
  float rotation_20;
  float rotation_21;
  float rotation_22;

  float translation_x;
  float translation_y;
  float translation_z;
};

struct Py2Cpp {
    char* template_list_path;
    float end00; float end01; float end02; float end03;
    float end10; float end11; float end12; float end13;
    float end20; float end21; float end22; float end23;
    float end30; float end31; float end32; float end33;
};

Var2Py var2py_ = {1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0};
Py2Cpp py2cpp_ = {nullptr, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

// extern Cpp declaration
extern "C"
{
  void TemplateAlign(char* target_pcd_path);

  Var2Py* get_var2py() { return &var2py_; };
  void set_py2cpp(Py2Cpp* vars) { py2cpp_ = *vars; };
}

// base tranforms before calculation
class StandardTrans
{
  public:
    Eigen::Matrix4f trans_camera2end;
    Eigen::Matrix4f trans_base2world;
    Eigen::Matrix4f trans_hole2world;
    Eigen::Matrix4f trans_end2base;

    Eigen::Matrix4f trans_hole2base;
    Eigen::Matrix4f trans_camera2base;
    Eigen::Matrix4f trans_hole2camera;

  StandardTrans(){
    trans_base2world << 1.0, 0.0, 0.0, -0.13133,
                        0.0, 1.0, 0.0, -0.201645,
                        0.0, 0.0, 1.0, 0.013512,
                        0.0, 0.0, 0.0, 1.0;
    
    trans_camera2end << 0.04999190841220458, 0.9987464691287659, -0.002509879704122885, -0.1233286407353617,
                        -0.9986450328192172, 0.05002288314324066, 0.01434606522856827, 0.003561782838864993,
                        0.01445363341206645, 0.001789291720513989, 0.9998939398337848, 0.0788226942731852,
                        0, 0, 0, 1;
                        
    trans_end2base << -0.5884786248207092, -2.325823970750207e-06, 0.8085127472877502, 0.8309828639030457, 
                      4.142234288906366e-08, 1.0, 2.9068187359371223e-06, -0.1501024067401886,
                      -0.8085127472877502, 1.744091264299641e-06, -0.5884786248207092, 0.2322056144475937, 
                      0.0, 0.0, 0.0, 1.0;

    float hole_position_x = 1.059;
    float hole_position_y = -0.313;
    float hole_position_z = 0.1134;
    float hole_orientation_x = 0.0;
    float hole_orientation_y = 0.0;
    float hole_orientation_z = 0.0;
    float hole_orientation_w = 1.0;
    
    trans_hole2world = Eigen::Matrix4f::Identity();
    Eigen::Quaternionf quat_hole2world(hole_orientation_w, hole_orientation_x, hole_orientation_y, hole_orientation_z);
    Eigen::Vector4f hole_pos_vec(hole_position_x, hole_position_y, hole_position_z, 1.0);
    trans_hole2world.block<3, 3>(0, 0) = quat_hole2world.matrix();
    trans_hole2world.block<4, 1>(0, 3) = hole_pos_vec;

    trans_hole2base = trans_base2world.inverse() * trans_hole2world;
    trans_camera2base = trans_end2base * trans_camera2end;
    trans_hole2camera = trans_camera2base.inverse() * trans_hole2base;
  }

  ~StandardTrans(){}
};

class FeatureCloud
{
    public:
      // rename some class
      typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
      typedef pcl::PointCloud<pcl::Normal> SurfaceNormal;
      typedef pcl::PointCloud<pcl::FPFHSignature33> LocalFeatures;
      typedef pcl::search::KdTree<pcl::PointXYZ> SearchMethod;

      //类内变量用下划线连接，并用下划线结尾
      //构造函数
      FeatureCloud():
        search_method_xyz_(new SearchMethod),
        normal_radius_(0.02f),
        feature_radius_(0.02f)
      {}

      //析构函数
      ~FeatureCloud(){}

      //设置点云的输入方式
      //直接输入一个点云指针
      void
      setInputCloud(PointCloud::Ptr xyz)
      {
        xyz_ = xyz;
        processInput();
      }

      //从pcd文件中加载点云
      void
      loadInputCloud(const std::string &pcd_file)
      {
        xyz_ = PointCloud::Ptr(new PointCloud);
        pcl::io::loadPCDFile(pcd_file, *xyz_);
        std::cout << xyz_->size() << std::endl;
        processInput();
      }

      //定义数据存取方法
      //读取指向点云的指针
      PointCloud::Ptr
      getPointCloud() const
      {
        return (xyz_);
      }

      //读取指向点云法线的指针
      SurfaceNormal::Ptr
      getSurfaceNormals() const
      {
        return (normals_);
      }

      //获取指向点云特征描述子的指针
      LocalFeatures::Ptr 
      getLocalFeatures() const
      {
        return (features_);
      }

    protected:
      //处理输入点云
      //计算点云曲面法线，估计曲面特征描述子
      void
      processInput()
      {
        computeSurfaceNormals();
        computerLocalFeatures();
      }

      void 
      computeSurfaceNormals()
      {
        normals_ = SurfaceNormal::Ptr(new SurfaceNormal);

        pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> norm_est;
        norm_est.setInputCloud(xyz_);
        norm_est.setSearchMethod(search_method_xyz_);
        norm_est.setRadiusSearch(normal_radius_);
        norm_est.compute(*normals_);
      }

      void 
      computerLocalFeatures()
      {
        features_ = LocalFeatures::Ptr(new LocalFeatures);

        pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_est;
        fpfh_est.setInputCloud(xyz_);
        fpfh_est.setInputNormals(normals_);
        fpfh_est.setSearchMethod(search_method_xyz_);
        fpfh_est.setRadiusSearch(feature_radius_);
        fpfh_est.compute(*features_);
      }
    
    private:
      //数据变量
      PointCloud::Ptr xyz_;
      SurfaceNormal::Ptr normals_;
      LocalFeatures::Ptr features_;
      SearchMethod::Ptr search_method_xyz_;

      //参数变量
      float normal_radius_;
      float feature_radius_;
};

class TemplateAlignment
{
    public:
      //创建一个储存匹配结果的结构
      struct Result
      {
        float fitness_score;
        Eigen::Matrix4f final_transformation;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
      };

      TemplateAlignment():
        min_sample_distance_(0.05f),
        max_correspondence_distance_(0.01f*0.01f),
        nr_iterations_(500)
        {}
      
      ~TemplateAlignment(){}

      // Set the given cloud as the target to which the templates will be aligned
      void
      setTargetCloud(FeatureCloud &target_cloud)
      {
        target_ = target_cloud;
      }

      // Add the given cloud to the list of template clouds
      void
      addTemplateCloud(FeatureCloud &template_cloud)
      {
        templates_.push_back(template_cloud);
      }

      // Align the given template cloud to the target specified by setTargetCloud ()
      void 
      align(FeatureCloud &template_cloud, TemplateAlignment::Result &result)
      {
        pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
        sac_ia.setMinSampleDistance(min_sample_distance_);
        sac_ia.setMaxCorrespondenceDistance(max_correspondence_distance_);
        sac_ia.setMaximumIterations(nr_iterations_);

        sac_ia.setInputTarget(target_.getPointCloud());
        sac_ia.setTargetFeatures(target_.getLocalFeatures());
        sac_ia.setInputCloud(template_cloud.getPointCloud());
        sac_ia.setSourceFeatures(template_cloud.getLocalFeatures());

        pcl::PointCloud<pcl::PointXYZ> registration_output;
        sac_ia.align(registration_output);

        result.fitness_score = (float) sac_ia.getFitnessScore(max_correspondence_distance_);
        result.final_transformation = sac_ia.getFinalTransformation();
      }

      // Align all of template clouds set by addTemplateCloud to the target specified by setTargetCloud ()
      void 
      alignAll(std::vector<TemplateAlignment::Result, Eigen::aligned_allocator<Result>> &results)
      {
        results.resize(templates_.size());
        tbb::parallel_for(0, int(templates_.size()), [&](int i)
        {
          align(templates_[i], results[i]);
        });
      }

      // Align all of template clouds to the target cloud to find the one with best alignment score
      int
      findBestAlignment(TemplateAlignment::Result &result)
      {
        // Align all of the templates to the target cloud
        std::vector<Result, Eigen::aligned_allocator<Result>> results;
        alignAll(results);

        // Find the template with the best (lowest) fitness score
        float lowest_score = std::numeric_limits<float>::infinity();
        int best_template = 0;
        for(size_t i = 0;i<results.size();i++)
        {
          const Result &r = results[i];
          if(r.fitness_score<lowest_score)
          {
            lowest_score = r.fitness_score;
            best_template = (int) i;
          }
        }

        result = results[best_template];
        return (best_template);
      }

      int
      findAlignment(TemplateAlignment::Result &result)
      {
        align(templates_[0], result);
        return 0;
      }

    private:
      // A list of template clouds and the target to which they will be aligned
      std::vector<FeatureCloud> templates_;
      FeatureCloud target_;

      // The Sample Consensus Initial Alignment (SAC-IA) registration routine and its parameters
      pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia_;
      float min_sample_distance_;
      float max_correspondence_distance_;
      int nr_iterations_;      
};

// visualization
boost::shared_ptr<pcl::visualization::PCLVisualizer> 
simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr template_cloud, pcl::PointCloud<pcl::PointXYZ>::ConstPtr target_cloud)
{
  boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Alignment Result"));
  viewer->setBackgroundColor(0,0,0);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> template_color(template_cloud,0,255,0);
  viewer->addPointCloud<pcl::PointXYZ>(template_cloud, template_color, "template cloud");
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud,255,0,0);
  viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "template cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
  viewer->addCoordinateSystem(0.1);
  viewer->setCameraPosition(0.0, 0.0, -1.0, 0, -1.0, 0.0);
  return (viewer);
}