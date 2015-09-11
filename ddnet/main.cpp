#include "./src/Solvers/BGD.h"
#include "./src/Solvers/ADAM.h"
#include "./src/Solvers/RMSPROP.h"
#include "./src/Network.h"
#include "./networks/MNIST.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <thrust/device_vector.h>

using namespace ddnet;
using namespace std;

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
		//SCALE DOWN!
		for(int i=0; i<1000; i++)
			cbuf[i] = cbuf[i]+1;
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

void run_mnist()
{
	//Require handle for operations	
	cudnnHandle_t handle = NULL;
	cudnnCreate(&handle);

	//Create network
	MNIST network(handle,{32});
	
	//Load MNSIT data set
	auto labels = load_mist_labels("./data/mnist/");
	auto images = load_mist_images("./data/mnist/");
	
	//Copy 128 images
	thrust::copy(images.begin(),images.begin()+32*28*28,network.input(0).data(0).begin());

	//Train the network while profiling
	for(int i=0; i<10000; i++)
	{
		//Forward network
		network.forward(0);

		cout << labels[0] << " " << network.output(0).data(0)[0] << endl;
		cout << labels[1] << " " << network.output(0).data(0)[1] << endl;
		cout << labels[2] << " " << network.output(0).data(0)[2] << endl;
		cout << endl << endl;

		//Set the class label
		set_target_values(labels,network.output(0),0);

		//Backward network
		// network.stream(0,0).backward(0);
		// network.stream(0,0).update(0);
		network.backward(0);
		network.update(0);

		getchar();
	}

	// Destroy handle
	cudnnDestroy(handle);
}

int main(int argc, char const *argv[])
{
	run_mnist();
}