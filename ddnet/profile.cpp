#include "./src/Solvers/ADAM.h"
#include "./src/Network.h"
#include "./networks/MNIST.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thrust/device_vector.h>

#include <cuda_profiler_api.h>
#include <nvToolsExt.h>

using namespace std;

void device_memory()
{
	size_t mem_free, mem_total;
	cudaMemGetInfo(&mem_free, & mem_total);
	cout << "Device Global Memory Usage: " << (mem_total-mem_free)/(1024*1024) << "[MB]/" << mem_total/(1024*1024) << "[MB]" << endl;
}

//Load the MNSIT data set from disk
thrust::device_vector<float> load_mist_labels(string file)
{
	/*READ LABELS*/
		//Open MNSIT data file for reading
		ifstream linf(file+"t10k-labels.idx1-ubyte", ios::binary);
		//Set exception mask
		linf.exceptions(ios::badbit | ios::failbit);
		//Skip 8 bytes
		linf.seekg(8);
		//Build temporary buffer
		array<unsigned char, 1000> buf;
		//Read labels
		linf.read((char*)buf.data(),1000);
		//Cast from unsigned char to float
		array<float, 1000> cbuf;
		copy(buf.begin(),buf.end(),cbuf.begin());
		//Construct device_vector
		thrust::device_vector<float> labels(cbuf.data(),cbuf.data()+1000);
	//Return labels
	return labels;
}

//Load the MNSIT data set from disk
thrust::device_vector<float> load_mist_images(string file)
{
	/*READ IMAGES*/
		//Open MNSIT data file for reading
		ifstream iinf(file+"t10k-images.idx3-ubyte", ios::binary);
		//Set exception mask
		iinf.exceptions(ios::badbit | ios::failbit);
		//Skip 16 bytes
		iinf.seekg(16);
		//Host vector buffer
		thrust::host_vector<float> ibuf(1000*28*28);
		//Variable to temporarily hold value
		char byte;
		//Fill host_vector
		for(int i=0; i<1000; i++)
			for(int r=0; r<28; r++)
				for(int c=0; c<28; c++)
				{
					iinf.read(&byte,1);
					ibuf[i*28*28+c*28+r] = float((unsigned char)byte)/255.0;
				}
		//Construct device_vector from host vector
		thrust::device_vector<float> images(ibuf);
	//Return images
	return images;
}

int main(int argc, char const *argv[])
{
	//Require handle for operations	
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	cout << "Before loading Network:" << endl; 
	device_memory();

	//Create network
	auto network = MNIST(handle,{32});

	//Print device memory
	cout << "After loading Network and before loading MNSIT:" << endl; 
	device_memory();

	//Load MNSIT data set
	auto labels = load_mist_labels("./data/mnist/");
	auto images = load_mist_images("./data/mnist/");
	
	//Print device memory
	cout << "After loading MNSIT:" << endl; 
	device_memory();

	//Copy 32 images
	thrust::copy(images.begin(),images.begin()+32*28*28,network.input(0).data(0).begin());

	//Train the network while profiling
	cudaProfilerStart(); nvtxRangePushA("Training");
	for(int i=0; i<2; i++)
	{
		nvtxRangePushA("EPOCH");
			//Forward network
			nvtxRangePushA("Forward");
				network.forward(0);
			nvtxRangePop();
			//Set the class label
			set_target_values(labels,network.output(0),0);
			//Backward network
			nvtxRangePushA("Backward");
				network.backward(0);
			nvtxRangePop();
			nvtxRangePushA("Update");
				network.update(0);
			nvtxRangePop();
		nvtxRangePop();
	}
	cudaProfilerStop(); nvtxRangePop();

	// Destroy handle
	cudnnDestroy(handle);
	
	return 0;
}