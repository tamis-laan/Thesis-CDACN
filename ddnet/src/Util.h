#pragma once
#include "cudnn.h"
#include <curand.h>
#include <cublas.h>
#include <string>

namespace ddnet
{
	//CUDNN EXCEPTIONS
	struct cudnn_exception : public std::exception
	{
		cudnnStatus_t error_;
		cudnn_exception(cudnnStatus_t error) : error_(error) {}
		virtual const char* what() const throw() { return cudnnGetErrorString(error_); }
	};
	inline void check(cudnnStatus_t error) { if(error) throw cudnn_exception(error); };

	//CUDA EXCEPTIONS
	struct cuda_exception : public std::exception
	{
		cudaError error_;
		cuda_exception(cudaError error) : error_(error) {}
		virtual const char* what() const throw() { return cudaGetErrorString(error_); }
	};
	inline void check(cudaError error) { if(error) throw cuda_exception(error); }

	//CURAND EXCEPTIONS
	struct curand_exception : public std::exception
	{
		curandStatus_t error_;
		curand_exception(curandStatus_t error) : error_(error) {}
		virtual const char* what() const throw() { return std::to_string(error_).c_str(); }
	};
	inline void check(curandStatus_t error) { if(error) throw curand_exception(error); }

	//CUBLAS EXCEPTIONS
	struct cublas_exception : public std::exception
	{
		cublasStatus_t error_;
		cublas_exception(cublasStatus_t error) : error_(error) {}
		virtual const char* what() const throw() { return std::to_string(error_).c_str(); }
	};
	inline void check(cublasStatus_t error) { if(error) throw cublas_exception(error); }

	//CUDNN WRAPPER EXCEPTIONS
	struct ddnet_exception : public std::exception
	{
		int error_;
		ddnet_exception(unsigned int error) : error_(error) {}
		virtual const char* what() const throw() 
		{ 
			switch(error_)
			{
				case 0:
					return "Input/Output dimention mismatch!";
				case 1:
					return "Probability must lie between 0 and 1!";
				case 2:
					return "Not enough Input/Output Blocks!";
				case 3:
					return "Reshape failed, dimention mismatch!";
				case 4:
					return "Unknown Block!!";
				case 5:
					return "Unknown Element!!";
				case 6:
					return "Using un-guarded Element!";
				case 7:
					return "Cannot create non Block/Element!";
				case 8:
					return "Cannot add none-type Element!";
				case 9:
					return "Cannot clone non type Block/Element!";
				case 10:
					return "Cannot Transfer Weights!";
				default:
					return "Generic Error!";
			}
		}
	};
}