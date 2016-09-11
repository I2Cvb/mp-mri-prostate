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

#include <itkMattesMutualInformationImageToImageMetricv4.h>

#include <itkIdentityTransform.h>
#include <itkCenteredTransformInitializer.h>
#include <itkVersorRigid3DTransform.h>

#include <itkLBFGSOptimizer.h>
#include <itkRegularStepGradientDescentOptimizerv4.h>

#include <itkMeanSquaresImageToImageMetric.h>

#include <itkRegularStepGradientDescentOptimizer.h>

#include <itkImageRegistrationMethodv4.h>
 
#include <itkLinearInterpolateImageFunction.h>

#include <itkResampleImageFilter.h>
#include <itkCastImageFilter.h>
 
#if ITK_VERSION_MAJOR >= 4
#include "gdcmUIDGenerator.h"
#else
#include "gdcm/src/gdcmFile.h"
#include "gdcm/src/gdcmUtil.h"
#endif

#include <algorithm>
#include <string>
#include <cstddef>

// Declare the function to copy DICOM dictionary
static void CopyDictionary(itk::MetaDataDictionary &fromDict, 
			   itk::MetaDataDictionary &toDict);

// Function to be used to sort the serie ID
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
    typedef double PixelType;
    typedef itk::Image< PixelType, in_dim > InImType;
    typedef itk::ImageSeriesReader< InImType > SeriesReader;
    typedef itk::ImageFileReader< InImType > FileReader;

    ///////////////////////////////////////////////////////////////////////////
    // Define the different transform used
    typedef double CoordinateRepType;
    // Auxiliary identity transform.
    typedef itk::IdentityTransform< double,
				    in_dim> IdentityTransformType;
    IdentityTransformType::Pointer identityTransform =
	IdentityTransformType::New();
    // Rigid transform
    typedef itk::VersorRigid3DTransform< double > TransformType;
        //Initialisation
    typedef itk::CenteredTransformInitializer<
	TransformType,
	InImType,
	InImType >  TransformInitializerType;
    // Define the optimizer
    typedef itk::RegularStepGradientDescentOptimizerv4< double > OptimizerType;

    // Define the metrics
    // Mutual Information Metric
    typedef itk::MattesMutualInformationImageToImageMetricv4 <
	InImType,
	InImType > MetricType;

    // Interpolation function
    typedef itk:: LinearInterpolateImageFunction< 
	InImType,
	CoordinateRepType > InterpolatorType;
 
    // Registration type
    typedef itk::ImageRegistrationMethodv4<
	InImType,
	InImType,
	TransformType>    RegistrationType;

    ///////////////////////////////////////////////////////////////////////////
    // Open all the information about the DCE information
    itk::GDCMSeriesFileNames::Pointer dce_gen =
	itk::GDCMSeriesFileNames::New();
    dce_gen->SetInputDirectory(argv[1]);

    // Get the serie UID
    const itk::SerieUIDContainer& dce_serie_uid = dce_gen->GetSeriesUIDs();
    // Sort the serie
    std::sort(dce_serie_uid.begin(), dce_serie_uid.end(), comparisonSerieID);

    // Read the serie which will be considered as fixed volume during
    // the registration
    const int serie_keep = 9;
    // Get the corresponding filenames
    const SeriesReader::FileNamesContainer& dce_fixed_filenames =
	dce_gen->GetFileNames(dce_serie_uid[serie_keep]);
    // Get the DICOM information and read the volume
    itk::GDCMImageIO::Pointer gdcm_dce_fixed = itk::GDCMImageIO::New();
    SeriesReader::Pointer dce_fixed_vol = SeriesReader::New();
    dce_fixed_vol->SetImageIO(gdcm_dce_fixed);
    dce_fixed_vol->SetFileNames(dce_fixed_filenames);

    // Try to read the volume
    try {
	dce_fixed_vol->Update();
    }
    catch(itk::ExceptionObject &excp) {
	std::cerr << "Exception thrown while reading the series"
		  << std::endl;
	std::cerr << excp << std::endl;
	return EXIT_FAILURE;
    }

    // Now, we have to read all the other series and they will be considered
    // as moving serie
    for (unsigned int nSerie = 0; nSerie < dce_serie_uid.size(); ++nSerie) {
	///////////////////////////////////////////////////////////////////////
	// Read the moving serie

	// Read the current serie
	const SeriesReader::FileNamesContainer& dce_moving_filenames = 
	    dce_gen->GetFileNames(dce_serie_uid[nSerie]);

        // Reader corresponding to the actual mask volume 
	itk::GDCMImageIO::Pointer gdcm_dce_moving = itk::GDCMImageIO::New();
	SeriesReader::Pointer dce_moving_vol = SeriesReader::New();
	dce_moving_vol->SetImageIO(gdcm_dce_moving);
        dce_moving_vol->SetFileNames(dce_moving_filenames);

	// Try to update to catch up any error
	try {
	    dce_moving_vol->Update();
	}
	catch(itk::ExceptionObject &excp) {
	    std::cerr << "Exception thrown while reading the series"
		      << std::endl;
	    std::cerr << excp << std::endl;
	    return EXIT_FAILURE;
	}

	// Continue only if the serie is not the fixed one
	if (nSerie == serie_keep) {
	    typedef signed short OutputPixelType;
	    const unsigned int out_dim = 2;
	    typedef itk::Image< OutputPixelType, out_dim > Image2DType;
	    typedef itk::ImageSeriesWriter< InImType,
					    Image2DType >  SeriesWriterType;

	    const char * outputDirectory = argv[2];
	    itksys::SystemTools::MakeDirectory(outputDirectory);
	    SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
	    gdcm_dce_moving->KeepOriginalUIDOn();
	    seriesWriter->SetInput(dce_moving_vol->GetOutput());
	    seriesWriter->SetImageIO(gdcm_dce_moving);
	    dce_gen->SetOutputDirectory(outputDirectory);
	    seriesWriter->SetFileNames(dce_gen->GetOutputFileNames());
	    seriesWriter->SetMetaDataDictionaryArray(
		dce_moving_vol->GetMetaDataDictionaryArray() );
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

	    continue;
	}


	MetricType::Pointer         metric        = MetricType::New();
	OptimizerType::Pointer      optimizer     = OptimizerType::New();
	RegistrationType::Pointer   registration  = RegistrationType::New();

	metric->SetNumberOfHistogramBins( 64 );
	metric->SetUseMovingImageGradientFilter( false );
	metric->SetUseFixedImageGradientFilter( false );
	metric->SetUseFixedSampledPointSet( false );
	
	registration->SetMetric(        metric        );
	registration->SetOptimizer( optimizer );

	TransformType::Pointer initialTransform = TransformType::New();

	registration->SetFixedImage(dce_fixed_vol->GetOutput());
	registration->SetMovingImage(dce_moving_vol->GetOutput());

	typedef itk::CenteredTransformInitializer<
	    TransformType,
	    InImType,
	    InImType >  TransformInitializerType;
	TransformInitializerType::Pointer initializer =
	    TransformInitializerType::New();

	initializer->SetTransform(   initialTransform );
	initializer->SetFixedImage(  dce_fixed_vol->GetOutput() );
	initializer->SetMovingImage( dce_moving_vol->GetOutput() );
	initializer->MomentsOn();
	initializer->InitializeTransform();

	typedef TransformType::VersorType  VersorType;
	typedef VersorType::VectorType     VectorType;
	VersorType     rotation;
	VectorType     axis;
	axis[0] = 0.0;
	axis[1] = 0.0;
	axis[2] = 1.0;
	const double angle = 0;
	rotation.Set(  axis, angle  );
	initialTransform->SetRotation( rotation );

	registration->SetInitialTransform( initialTransform );
	typedef OptimizerType::ScalesType       OptimizerScalesType;
	OptimizerScalesType optimizerScales( initialTransform->GetNumberOfParameters() );
	const double translationScale = 1.0 / 1000.0;
	optimizerScales[0] = 1.0;
	optimizerScales[1] = 1.0;
	optimizerScales[2] = 1.0;
	optimizerScales[3] = translationScale;
	optimizerScales[4] = translationScale;
	optimizerScales[5] = translationScale;
	optimizer->SetScales( optimizerScales );
	optimizer->SetNumberOfIterations( 200 );
	optimizer->SetLearningRate( 0.2 );
	optimizer->SetMinimumStepLength( 0.001 );
	optimizer->SetReturnBestParametersAndValue(true);
	// optimizer->MaximizeOn();

	RegistrationType::ShrinkFactorsArrayType shrinkFactorsPerLevel;
	shrinkFactorsPerLevel.SetSize( 1 );
	shrinkFactorsPerLevel[0] = 1;

	RegistrationType::SmoothingSigmasArrayType smoothingSigmasPerLevel;
	smoothingSigmasPerLevel.SetSize( 1 );
	smoothingSigmasPerLevel[0] = 0;

	const unsigned int numberOfLevels = 1;
	registration->SetNumberOfLevels( numberOfLevels );
	registration->SetSmoothingSigmasPerLevel( smoothingSigmasPerLevel );
	registration->SetShrinkFactorsPerLevel( shrinkFactorsPerLevel );

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

	const TransformType::ParametersType finalParameters =
	    registration->GetOutput()->Get()->GetParameters();

	const double versorX              = finalParameters[0];
	const double versorY              = finalParameters[1];
	const double versorZ              = finalParameters[2];
	const double finalTranslationX    = finalParameters[3];
	const double finalTranslationY    = finalParameters[4];
	const double finalTranslationZ    = finalParameters[5];
	const unsigned int numberOfIterations = optimizer->GetCurrentIteration();
	const double bestValue = optimizer->GetValue();

	// Print out results
	//
	std::cout << std::endl << std::endl;
	std::cout << "Result = " << std::endl;
	std::cout << " versor X      = " << versorX  << std::endl;
	std::cout << " versor Y      = " << versorY  << std::endl;
	std::cout << " versor Z      = " << versorZ  << std::endl;
	std::cout << " Translation X = " << finalTranslationX  << std::endl;
	std::cout << " Translation Y = " << finalTranslationY  << std::endl;
	std::cout << " Translation Z = " << finalTranslationZ  << std::endl;
	std::cout << " Iterations    = " << numberOfIterations << std::endl;
	std::cout << " Metric value  = " << bestValue << std::endl;

	TransformType::Pointer finalTransform = TransformType::New();

	finalTransform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
	finalTransform->SetParameters( finalParameters );

	// Software Guide : BeginCodeSnippet
	TransformType::MatrixType matrix = finalTransform->GetMatrix();
	TransformType::OffsetType offset = finalTransform->GetOffset();
	std::cout << "Matrix = " << std::endl << matrix << std::endl;
	std::cout << "Offset = " << std::endl << offset << std::endl;
	
	// ///////////////////////////////////////////////////////////////////////
	// RigidTransformType::Pointer initialTransform = RigidTransformType::New();
	// TransformInitializerType::Pointer initializer =
	//     TransformInitializerType::New();
	// initializer->SetTransform(initialTransform);
	// initializer->SetFixedImage(dce_fixed_vol->GetOutput());
	// initializer->SetMovingImage(dce_moving_vol->GetOutput());
	// initializer->MomentsOn();
	// initializer->InitializeTransform();
	// typedef RigidTransformType::VersorType VersorType;
	// typedef VersorType::VectorType VectorType;
	// VersorType rotation;
	// VectorType axis;
	// axis[0] = 0.0;
	// axis[1] = 0.0;
	// axis[2] = 1.0;
	// const double angle = 0;
	// rotation.Set(axis, angle);
	// initialTransform->SetRotation(rotation);
	// // Find the rigid transformation to apply
	// RigidTransformType::Pointer rigid_transform =
	//     RigidTransformType::New();

	// // Define the registration
	// M2MetricType::Pointer metric = M2MetricType::New();
	// GDOptimizerType::Pointer gd_optimizer = GDOptimizerType::New();
	InterpolatorType::Pointer interpolator = InterpolatorType::New();
	// RegistrationType::Pointer registration = RegistrationType::New();
	// registration->SetInitialTransformParameters( initialTransform->GetParameters() );
	// registration->SetMetric(metric);
	// registration->SetOptimizer(gd_optimizer);
	// registration->SetInterpolator(interpolator);

	// // Set-up the input images
	// registration->SetFixedImage(dce_fixed_vol->GetOutput());
	// registration->SetMovingImage(dce_moving_vol->GetOutput());

	// // Initialise the transform
	// registration->SetFixedImageRegion(
	//     dce_fixed_vol->GetOutput()->GetLargestPossibleRegion());
	// // registration->SetTransform(rigid_transform);

	// // Initialize with default parameters
	// // typedef RegistrationType::ParametersType ParametersType;
	// // ParametersType init_params(rigid_transform->GetNumberOfParameters());
	// // init_params[0] = 0.;
	// // init_params[1] = 0.;
	// // init_params[2] = 0.;
	// // init_params[3] = 0.;
	// // init_params[4] = 0.;
	// // init_params[5] = 0.;
	// // registration->SetInitialTransformParameters(init_params);

	// //  Define optimizer normaliztion to compensate for different dynamic
	// // range of rotations and translations.
	// typedef GDOptimizerType::ScalesType OptimizerScalesType;
	// OptimizerScalesType optimizer_scales(
	//     rigid_transform->GetNumberOfParameters());
	// const double translation_scale = 1.0 / 1000.0;
	// optimizer_scales[0] = 1.0;
	// optimizer_scales[1] = 1.0;
	// optimizer_scales[2] = 1.0;
	// optimizer_scales[3] = translation_scale;
	// optimizer_scales[4] = translation_scale;
	// optimizer_scales[5] = translation_scale;
	// gd_optimizer->SetScales(optimizer_scales);

	// gd_optimizer->SetMaximumStepLength(0.2000);
	// gd_optimizer->SetMinimumStepLength(0.0001);
	// gd_optimizer->SetNumberOfIterations(1000);
	// gd_optimizer->MaximizeOn();

	// InImType::RegionType fixedImageRegion =
	//     dce_fixed_vol->GetOutput()->GetLargestPossibleRegion();
	// const unsigned int numberOfPixels =
	//     fixedImageRegion.GetNumberOfPixels();
	// const unsigned int numberOfSamples =
	//     static_cast< unsigned int >(numberOfPixels * 0.006); 
	// metric->SetNumberOfSpatialSamples(numberOfSamples);
	// metric->SetNumberOfThreads(8);
	// metric->SetNumberOfHistogramBins(60);

	// std::cout << "Starting Rigid Registration " << std::endl;
	// try
	// {
	//     registration->Update();
	//     std::cout << "Optimizer stop condition = "
	// 	      << registration->GetOptimizer()->GetStopConditionDescription()
	// 	      << std::endl;
	// }
	// catch(itk::ExceptionObject & err)
	// {
	//     std::cerr << "ExceptionObject caught !" << std::endl;
	//     std::cerr << err << std::endl;
	//     return EXIT_FAILURE;
	// }
	// std::cout << "Rigid Registration completed" << std::endl;

	// GDOptimizerType::ParametersType finalParameters =
	//     registration->GetLastTransformParameters();
	// std::cout << "Last Transform Parameters" << std::endl;
	// std::cout << finalParameters << std::endl;
	// // rigid_transform->SetParameters(finalParameters);

	// rigid_transform->SetFixedParameters( registration->GetOutput()->Get()->GetFixedParameters() );
	// rigid_transform->SetParameters( finalParameters );

	// Resample the data with the transform found previously
	typedef itk::ResampleImageFilter< InImType,
					  InImType > ResampleFilterType;
	ResampleFilterType::Pointer resample = ResampleFilterType::New();
	resample->SetTransform(finalTransform);
	resample->SetInput(dce_moving_vol->GetOutput());
	resample->SetSize(
	    dce_moving_vol->GetOutput()->GetLargestPossibleRegion().GetSize());
	resample->SetInterpolator(interpolator);
	resample->SetOutputOrigin(dce_moving_vol->GetOutput()->GetOrigin());
	resample->SetOutputSpacing(dce_moving_vol->GetOutput()->GetSpacing());
	resample->SetOutputDirection(
	    dce_moving_vol->GetOutput()->GetDirection());
	resample->SetDefaultPixelValue(0);

	typedef signed short OutputPixelType;
	const unsigned int out_dim = 2;
	typedef itk::Image< OutputPixelType, out_dim > Image2DType;
	typedef itk::ImageSeriesWriter< InImType,
					Image2DType >  SeriesWriterType;

	const char * outputDirectory = argv[2];
	itksys::SystemTools::MakeDirectory(outputDirectory);
	SeriesWriterType::Pointer seriesWriter = SeriesWriterType::New();
	gdcm_dce_moving->KeepOriginalUIDOn();
	seriesWriter->SetInput(resample->GetOutput());
	seriesWriter->SetImageIO(gdcm_dce_moving);
	dce_gen->SetOutputDirectory(outputDirectory);
	seriesWriter->SetFileNames(dce_gen->GetOutputFileNames());
	seriesWriter->SetMetaDataDictionaryArray(
	    dce_moving_vol->GetMetaDataDictionaryArray() );
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
    }

    return EXIT_SUCCESS;
}
