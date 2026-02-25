I created another project folder "benchmark_engine" inside "example" and a "morpheus_benchmark_engine" folder in parallel to the "morpheus_dfp" folder. 

The goal is to test if we can swap model architecture. 
We going to implement a BenchmarkEngineTrainingBase class, which replaces the DFPTraining class, having the on_data() function. Inside on_data(), we implement a `_create_model()` function that determines whether we use AutoEncoder or ResidualAutoEncoder. 

We make a AutoEncoderTraining class and a ResidualAutoEncoderTraining class, inheriting the BenchmarkEngineTrainingBase class, which implements the `_create_model()` function and instantiates the model classes, similar to the AutoEncoder class (having the fit() function). 

Next, we implement the specific architecture classes like AEModule and ResidualAEModule which have the `forward()` function. 

The experiment is this:
Compare the runtime of the two scenarios:
1. Run AutoEncoder and ResidualAutoEncoder in two separate linear pipelines
	1. Create two python files (by duplicating and modifying dfp_duo_pipeline.py) and replace the DFPTraining stage with the two new Training stage classes, respectively
2. Create a nonlinear pipeline by connecting the last data preprocessing stage to both AutoEncoder and ResidualAutoEncoder and let the Morpheus pipeline figure out how to allocate the resource. 

Next, we copy dfp_duo_pipeline.py into another python file called "dfp_duo_"
Now inside the run_pipeline() in dfp_duo_pipeline.py, we create a pipeline, that connects data to both "AutoEncoderTraining" and "ResidualAutoEncoderTraining".




