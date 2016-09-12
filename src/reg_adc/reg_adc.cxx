#include <itkVersion.h>

#include <itkImage.h>
#include <itkStatisticsImageFilter.h>
 
#include <itkGDCMImageIO.h>
#include <itkGDCMSeriesFileNames.h>
#include <itkNumericSeriesFileNames.h>
#include <gdcmUIDGenerator.h>
#include <itkNiftiImageIO.h>
 
#include <itkImageSeriesReader.h>
#include <itkImageSeriesWriter.h>

#include <itkIdentityTransform.h>
#include <itkVersorRigid3DTransform.h>
#include <itkBSplineTransform.h>

#include <itkLBFGSOptimizer.h>
#include <itkRegularStepGradientDescentOptimizer.h>

#include <itkMeanSquaresImageToImageMetric.h>

#include <itkImageRegistrationMethod.h>

#include <itkBSplineResampleImageFunction.h>
#include <itkBSplineDecompositionImageFilter.h>

#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>

#include <algorithm>
#include <string>
#include <cstddef>


static void CopyDictionary (itk::MetaDataDictionary &fromDict,
			    itk::MetaDataDictionary &toDict);

bool comparisonSerieID(std::string i, std::string j) {
    // We have to extract the number after the last point
    std::size_t found_last_point = i.find_last_of(".");
    // Convert string to integer for the last number of the chain
    int i_int = std::stoi(i.substr(found_last_point+1));
    // We have to extract the number after the last point
    found_last_point = j.find_last_of(".");
    // Convert string to integer for the last number of the chain
    int j_int = std::stoi(j.substr(found_last_point+1));
    return i_int < j_int;
}

int main( int argc, char* argv[] )
{

    ///////////////////////////////////////////////////////////////////////////
    // Define the image type
    const unsigned int in_dim = 3; 
    typedef short signed PixelType;
    typedef itk::Image< PixelType, in_dim > InImType;
    typedef itk::ImageSeriesReader< InImType > SeriesReader;
    typedef itk::ImageFileReader< InImType > FileReader;

    ///////////////////////////////////////////////////////////////////////////
    // Define the different transform used
    typedef double CoordinateRepType;
    const unsigned int space_dim = in_dim;
    // Auxiliary identity transform.
    typedef itk::IdentityTransform< double,
				    in_dim> IdentityTransformType;
    IdentityTransformType::Pointer identityTransform =
	IdentityTransformType::New();
    // Rigid transform
    typedef itk::VersorRigid3DTransform< double > RigidTransformType;
    // BSpline transform
    const unsigned int spline_order = 3;
    typedef itk::BSplineTransform< CoordinateRepType,
				   space_dim,
				   spline_order > DeformableTransformType;

    // Define the optimizer
    typedef itk::RegularStepGradientDescentOptimizer GDOptimizerType;

    // Define the metrics
    // Mean squares error metri
    typedef itk::MeanSquaresImageToImageMetric< InImType,
						InImType > M2MetricType;

    // Interpolation function
    typedef itk:: LinearInterpolateImageFunction< 
	InImType,
	CoordinateRepType > InterpolatorType;
 
    // Registration type
    typedef itk::ImageRegistrationMethod<
	InImType,
	InImType >    RegistrationType;

    ///////////////////////////////////////////////////////////////////////////
    // Read the image

    // T2W GT - DICOM format
    itk::GDCMSeriesFileNames::Pointer gt_t2w_gen =
	itk::GDCMSeriesFileNames::New();
    gt_t2w_gen->SetInputDirectory(argv[1]);
    const SeriesReader::FileNamesContainer& gt_t2w_filenames =
	gt_t2w_gen->GetInputFileNames();

    itk::GDCMImageIO::Pointer gdcm_gt_t2w = itk::GDCMImageIO::New();
    SeriesReader::Pointer gt_t2w = SeriesReader::New();
    gt_t2w->SetImageIO(gdcm_gt_t2w);
    gt_t2w->SetFileNames(gt_t2w_filenames);

    try {
	gt_t2w->Update();
    }
    catch (itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series" << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    // Print information about the volume
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the gt_t2w:" << std::endl;
    std::cout << "Spacing: " << gt_t2w->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << gt_t2w->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:" <<
	std::endl << gt_t2w->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:"
	      << gt_t2w->GetOutput()->GetLargestPossibleRegion().GetSize()
	      << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;


    // ADC GT - NIFTI format
    FileReader::Pointer gt_adc = FileReader::New();
    gt_adc->SetFileName(argv[2]);
    gt_adc->Update();

    // Print information about the volume
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the gt_adc:" << std::endl;
    std::cout << "Spacing: " << gt_adc->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << gt_adc->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:"
	      << std::endl << gt_adc->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:"
	      << gt_adc->GetOutput()->GetLargestPossibleRegion().GetSize()
	      << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    ///////////////////////////////////////////////////////////////////////////
    // INITIALISATION + 3D RIGID REGISTRATION
    RigidTransformType::Pointer rigid_transform =
	RigidTransformType::New();

    // Define the registration
    M2MetricType::Pointer metric = M2MetricType::New();
    GDOptimizerType::Pointer gd_optimizer = GDOptimizerType::New();
    InterpolatorType::Pointer interpolator = InterpolatorType::New();
    RegistrationType::Pointer registration = RegistrationType::New();
    registration->SetMetric(metric);
    registration->SetOptimizer(gd_optimizer);
    registration->SetInterpolator(interpolator);

    // Set-up the input images
    registration->SetFixedImage(gt_t2w->GetOutput());
    registration->SetMovingImage(gt_adc->GetOutput());

    // Initialise the transform
    registration->SetFixedImageRegion(
	gt_t2w->GetOutput()->GetLargestPossibleRegion());
    registration->SetTransform(rigid_transform);

    // Initialize with default parameters
    typedef RegistrationType::ParametersType ParametersType;
    ParametersType init_params(rigid_transform->GetNumberOfParameters());
    init_params[0] = 0.;
    init_params[1] = 0.;
    init_params[2] = 0.;
    init_params[3] = 0.;
    init_params[4] = 0.;
    init_params[5] = 0.;
    registration->SetInitialTransformParameters(init_params);

    //  Define optimizer normaliztion to compensate for different dynamic range
    //  of rotations and translations.
    typedef GDOptimizerType::ScalesType OptimizerScalesType;
    OptimizerScalesType optimizer_scales(
    	rigid_transform->GetNumberOfParameters());
    const double translation_scale = 1.0 / 1000.0;
    optimizer_scales[0] = 1.0;
    optimizer_scales[1] = 1.0;
    optimizer_scales[2] = 1.0;
    optimizer_scales[3] = translation_scale;
    optimizer_scales[4] = translation_scale;
    optimizer_scales[5] = translation_scale;
    gd_optimizer->SetScales(optimizer_scales);

    gd_optimizer->SetMaximumStepLength(0.2000);
    gd_optimizer->SetMinimumStepLength(0.0001);
    gd_optimizer->SetNumberOfIterations(200);

    metric->SetNumberOfSpatialSamples(10000L);
    
    std::cout << "Starting Rigid Registration " << std::endl;
    try
    {
	registration->Update();
	std::cout << "Optimizer stop condition = "
		  << registration->GetOptimizer()->GetStopConditionDescription()
		  << std::endl;
    }
    catch(itk::ExceptionObject & err)
    {
	std::cerr << "ExceptionObject caught !" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
    }
    std::cout << "Rigid Registration completed" << std::endl;

    GDOptimizerType::ParametersType finalParameters =
    	registration->GetLastTransformParameters();
    std::cout << "Last Transform Parameters" << std::endl;
    std::cout << finalParameters << std::endl;

    // BSPLINE REGISTRATION
    // Coarse optimization
    DeformableTransformType::Pointer bspline_coarse_transform =
	DeformableTransformType::New();
    DeformableTransformType::PhysicalDimensionsType fixed_phy_dim;
    DeformableTransformType::MeshSizeType mesh_size;
    DeformableTransformType::OriginType fixed_origin;
    unsigned int nb_nodes_coarse = 5;

    for(unsigned int i = 0; i< space_dim; ++i)
    {
	fixed_origin[i] = gt_t2w->GetOutput()->GetOrigin()[i];
	fixed_phy_dim[i] = gt_t2w->GetOutput()->GetSpacing()[i] *
	    static_cast<double>(
		gt_t2w->GetOutput()->GetLargestPossibleRegion().GetSize()[i] -
		1);
    }
    mesh_size.Fill(nb_nodes_coarse - spline_order);
    bspline_coarse_transform->SetTransformDomainOrigin(fixed_origin);
    bspline_coarse_transform->SetTransformDomainPhysicalDimensions(
	fixed_phy_dim);
    bspline_coarse_transform->SetTransformDomainMeshSize(mesh_size);
    bspline_coarse_transform->SetTransformDomainDirection(
	gt_t2w->GetOutput()->GetDirection() );
    typedef DeformableTransformType::ParametersType ParametersType;
    unsigned int nb_spline_params =
	bspline_coarse_transform->GetNumberOfParameters();
    optimizer_scales = OptimizerScalesType(nb_spline_params);
    optimizer_scales.Fill(1.0);
    gd_optimizer->SetScales(optimizer_scales);
    ParametersType init_bspline_params(nb_spline_params);
    init_bspline_params.Fill( 0.0 );
    bspline_coarse_transform->SetParameters(init_bspline_params);
    registration->SetInitialTransformParameters(
	bspline_coarse_transform->GetParameters());
    registration->SetTransform(bspline_coarse_transform);

    gd_optimizer->SetMaximumStepLength(10.0);
    gd_optimizer->SetMinimumStepLength(0.01);
    gd_optimizer->SetRelaxationFactor(0.7);
    gd_optimizer->SetNumberOfIterations(50);

    metric->SetNumberOfSpatialSamples(nb_spline_params * 100);

    std::cout << std::endl << "Starting Deformable Registration Coarse Grid"
	      << std::endl;
    try
    {
	registration->Update();
    }
    catch(itk::ExceptionObject & err)
    {
	std::cerr << "ExceptionObject caught !" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
    }
    std::cout << "Deformable Registration Coarse Grid completed" << std::endl;
    std::cout << std::endl;

    finalParameters = registration->GetLastTransformParameters();
    bspline_coarse_transform->SetParameters(finalParameters);

    std::cout << "Last Transform Parameters" << std::endl;
    std::cout << finalParameters << std::endl;

    // Fine optimization
    DeformableTransformType::Pointer bspline_fine_transform =
	DeformableTransformType::New();
    unsigned int nb_nodes_fine = 20;
    mesh_size.Fill(nb_nodes_fine - spline_order);
    bspline_fine_transform->SetTransformDomainOrigin(fixed_origin);
    bspline_fine_transform->SetTransformDomainPhysicalDimensions(
	fixed_phy_dim);
    bspline_fine_transform->SetTransformDomainMeshSize(mesh_size);
    bspline_fine_transform->SetTransformDomainDirection(
	gt_t2w->GetOutput()->GetDirection());
    nb_spline_params = bspline_fine_transform->GetNumberOfParameters();
    ParametersType params_high(nb_spline_params);
    params_high.Fill(0.0);

    unsigned int counter = 0;
    for (unsigned int k = 0; k < space_dim; ++k)
    {
	typedef DeformableTransformType::ImageType ParametersImageType;
	typedef itk::ResampleImageFilter< ParametersImageType,
					  ParametersImageType > ResamplerType;
	ResamplerType::Pointer upsampler = ResamplerType::New();
	typedef itk::BSplineResampleImageFunction< ParametersImageType,
						   double > FunctionType;
	FunctionType::Pointer function = FunctionType::New();
	upsampler->SetInput(
	    bspline_coarse_transform->GetCoefficientImages()[k]);
	upsampler->SetInterpolator(function);
	upsampler->SetTransform(identityTransform);
	upsampler->SetSize(bspline_fine_transform->GetCoefficientImages()[k]->
			    GetLargestPossibleRegion().GetSize());
	upsampler->SetOutputSpacing(
	    bspline_fine_transform->GetCoefficientImages()[k]->GetSpacing());
	upsampler->SetOutputOrigin(
	    bspline_fine_transform->GetCoefficientImages()[k]->GetOrigin());
	typedef itk::BSplineDecompositionImageFilter< ParametersImageType,
						      ParametersImageType >
	    DecompositionType;
	DecompositionType::Pointer decomposition = DecompositionType::New();
	decomposition->SetSplineOrder(spline_order);
	decomposition->SetInput(upsampler->GetOutput());
	decomposition->Update();
	ParametersImageType::Pointer newCoefficients =
	    decomposition->GetOutput();
	// copy the coefficients into the parameter array
	typedef itk::ImageRegionIterator<ParametersImageType> Iterator;
	Iterator it(newCoefficients,
		    bspline_fine_transform->GetCoefficientImages()[k]->
		    GetLargestPossibleRegion());
	while (!it.IsAtEnd())
	{
	    params_high[counter++] = it.Get();
	    ++it;
	}
    }
    optimizer_scales = OptimizerScalesType(nb_spline_params);
    optimizer_scales.Fill(1.0);
    gd_optimizer->SetScales(optimizer_scales);
    bspline_fine_transform->SetParameters(params_high);

    std::cout << "Starting Registration with high resolution transform"
	      << std::endl;

    registration->SetInitialTransformParameters(
	bspline_fine_transform->GetParameters());
    registration->SetTransform(bspline_fine_transform);

    InImType::RegionType fixedRegion =
	gt_t2w->GetOutput()->GetLargestPossibleRegion();
    const unsigned int numberOfPixels = fixedRegion.GetNumberOfPixels();

    const unsigned long numberOfSamples =
	static_cast<unsigned long>(
	    std::sqrt(static_cast<double>(nb_spline_params) *
		       static_cast<double>(numberOfPixels)));
    metric->SetNumberOfSpatialSamples(numberOfSamples);
    try
    {
	registration->Update();
    }
    catch(itk::ExceptionObject & err)
    {
	std::cerr << "ExceptionObject caught !" << std::endl;
	std::cerr << err << std::endl;
	return EXIT_FAILURE;
    }
    std::cout << "Deformable Registration Fine Grid completed" << std::endl;
    std::cout << std::endl;

    finalParameters = registration->GetLastTransformParameters();
    bspline_fine_transform->SetParameters(finalParameters);

    std::cout << "Last Transform Parameters" << std::endl;
    std::cout << finalParameters << std::endl;

    // The transform was found and we can now read the data and then
    // transform each ADC volume

    ///////////////////////////////////////////////////////////////////////////
    // Get the information regarding the ADC series
    itk::GDCMSeriesFileNames::Pointer adc_gen =
	itk::GDCMSeriesFileNames::New();
    adc_gen->SetInputDirectory(argv[3]);
    const SeriesReader::FileNamesContainer& adc_filenames =
	adc_gen->GetInputFileNames();

    itk::GDCMImageIO::Pointer gdcm_adc = itk::GDCMImageIO::New();
    SeriesReader::Pointer adc = SeriesReader::New();
    adc->SetImageIO(gdcm_adc);
    adc->SetFileNames(adc_filenames);

    try {
	adc->Update();
    }
    catch (itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series" << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    // Print information about the volume
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "Information about the adc:" << std::endl;
    std::cout << "Spacing: " << adc->GetOutput()->GetSpacing() << std::endl;
    std::cout << "Origin:" << adc->GetOutput()->GetOrigin() << std::endl;
    std::cout << "Direction:" <<
	std::endl << adc->GetOutput()->GetDirection() << std::endl;
    std::cout << "Size:"
	      << adc->GetOutput()->GetLargestPossibleRegion().GetSize()
	      << std::endl;
    std::cout << "" << std::endl;
    std::cout << "******************************" << std::endl;
    std::cout << "" << std::endl;

    // Resample the data with the transform found previously
    typedef itk::ResampleImageFilter< InImType,
				      InImType > ResampleFilterType;
    ResampleFilterType::Pointer resample = ResampleFilterType::New();
    InterpolatorType::Pointer interpolator2 = InterpolatorType::New();
    resample->SetTransform(bspline_fine_transform);
    resample->SetInput(adc->GetOutput());
    resample->SetSize(
	adc->GetOutput()->GetLargestPossibleRegion().GetSize());
    resample->SetInterpolator(interpolator2);
    resample->SetOutputOrigin(adc->GetOutput()->GetOrigin());
    resample->SetOutputSpacing(adc->GetOutput()->GetSpacing());
    resample->SetOutputDirection(adc->GetOutput()->GetDirection());
    resample->SetDefaultPixelValue(0);

    typedef signed short OutputPixelType;
    const unsigned int out_dim = 2;
    typedef itk::Image< OutputPixelType, out_dim > Image2DType;
    typedef itk::ImageSeriesWriter< InImType,
				    Image2DType >  SeriesWriterType;

    const char * outputDirectory = argv[4];
    itksys::SystemTools::MakeDirectory(outputDirectory);
    SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
    gdcm_adc->KeepOriginalUIDOn();
    seriesWriter->SetInput(resample->GetOutput());
    seriesWriter->SetImageIO(gdcm_adc);
    adc_gen->SetOutputDirectory(outputDirectory);
    seriesWriter->SetFileNames(adc_gen->GetOutputFileNames());
    seriesWriter->SetMetaDataDictionaryArray(
	adc->GetMetaDataDictionaryArray() );
    try
    {
	seriesWriter->Update();
    }
    catch(itk::ExceptionObject & excp)
    {
	std::cerr << "Exception thrown while writing the series "
		  << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    std::cout << " Done!" << std::endl;

    return EXIT_SUCCESS;
}

void CopyDictionary (itk::MetaDataDictionary &fromDict,
		     itk::MetaDataDictionary &toDict)
{
    typedef itk::MetaDataDictionary DictionaryType;
 
    DictionaryType::ConstIterator itr = fromDict.Begin();
    DictionaryType::ConstIterator end = fromDict.End();
    typedef itk::MetaDataObject< std::string > MetaDataStringType;
 
    while( itr != end )
    {
	itk::MetaDataObjectBase::Pointer  entry = itr->second;
 
	MetaDataStringType::Pointer entryvalue = 
	    dynamic_cast<MetaDataStringType *>( entry.GetPointer() ) ;
	if( entryvalue )
	{
	    std::string tagkey   = itr->first;
	    std::string tagvalue = entryvalue->GetMetaDataObjectValue();
	    itk::EncapsulateMetaData<std::string>(toDict, tagkey, tagvalue); 
	    //std::cout << tagkey << " " << tagvalue << std::endl;
	}
	++itr;
    }
}
