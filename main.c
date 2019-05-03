#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "lodepng.h"

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

#include <CL/cl.h>
#include <Windows.h>


const int resizear = 4;    // valor por el que dividimos la imagen
const uint32_t half_winx = 8;
const uint32_t half_winy = 15;
const int threshold = 8;    // Threshold para calcular el mapa final
const uint32_t win_area = 527;  // (half_winx x 2 + 1) x (half_winY x 2 + 1) = win_area

int maxDisp = 64;
int minDisp = 0;
/*
typedef struct cl_image_desc {
	cl_mem_object_type image_type;
	size_t image_width;
	size_t image_height;
	size_t image_depth;
	size_t image_array_size;
	size_t image_row_pitch;
	size_t image_slice_pitch;
	cl_uint num_mip_levels;
	cl_uint num_samples;
	cl_mem buffer;
} cl_image_desc;
*/
cl_image_desc imgDescriptor;        // Image descriptor contains type and dimensions of the image
cl_image_format format = { CL_RGBA, CL_UNSIGNED_INT8 };


char *leerFıchero(const char *filename);
cl_kernel file_to_kernel(cl_context ctx, char const *kernel, char const *kernel_name);
void normalizar(uint8_t* dispMap, uint32_t w, uint32_t h);
uint8_t* SustituirCeros(const uint8_t* dispMap, uint32_t w, uint32_t h);


int32_t main(){

	static LARGE_INTEGER frequencyStart, frequencyEnd; //variables para medir lo que tarda el programa en ejecutar
	LARGE_INTEGER starTime, endTime;

	if (frequencyStart.QuadPart == 0)
		::QueryPerformanceFrequency(&frequencyStart);
	::QueryPerformanceCounter(&starTime);

	printf("Comenzando programa, esto tardará unos minutos...\n");

	uint8_t *image0, *image1; // Imagenes iniciales
	uint8_t *dDisparity, *Disparity;

	uint32_t err;                       // Error code, 0 is OK
	uint32_t new_width, new_height;     // nuevo tamaño
	uint32_t w0, h0, w1, h1;            // tamaños originales


	cl_context ctx;
	cl_command_queue queue;
	cl_int status;

	const size_t localWorkSize[] = { 4, 5 };       // Local work size
	  //const size_t localWorkSize[]    = {2, 2};       // Local work size (Odroid)
	const size_t globalWorkSize[] = { 504, 735 };   // Global work size

	const size_t localWorkSize1D[] = { localWorkSize[0] * localWorkSize[1] };      // 1-dimentional local work size
	const size_t globalWorkSize1D[] = { globalWorkSize[0] * globalWorkSize[1] };    // 1-dimentional global work size


	// CARGAR IMAGENES EN MEMORIA Y COMPROBAR POSIBLES ERRORES
	lodepng_decode32_file(&image0, &w0, &h0, "im0.png");  //CARGANDO IMAGEN DERECHA
	lodepng_decode32_file(&image1, &w1, &h1, "im1.png");  //CARGANDO IMAGEN IZQUIERDA
	
	if (w0 != w1 || h0 != h1) {
		printf("Error, tamaños diferentes.\n");
		free(image0);
		free(image1);
		return -1;
	}
	new_width = w0 / resizear;
	new_height = h0 / resizear;



	// ******** Setup OpenCL environment to run the kernel ********
	cl_platform_id platform = 0;
	cl_device_id device = 0;
	cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };

	status = clGetPlatformIDs(1, &platform, NULL);
	printf("clGetPlatformIDs status == CL_SUCCESS - %d\n", status == CL_SUCCESS);
	int gpu = 1; // O : CPU, 1 : GPU
	status = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);

	props[1] = (cl_context_properties)platform;
	// context
	ctx = clCreateContext(props, 1, &device, NULL, NULL, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Fail to create context for OpenCL !\n");
		abort();
	}
	// queue
	queue = clCreateCommandQueue(ctx, device, 0, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Fail to create queue for OpenCL context !\n");
		abort();
	}


	// MEMORIA PARA IMAGENES Y DISPARITY MAPS
	cl_mem clmemImage0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status); //CREAR BUFFER IMAGEN DERECHA
	cl_mem clmemImage1 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status); //CREAR BUFFER IMAGEN IZQUIERDA

	cl_mem IzqDer = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status); //CREAR BUFFER PARA DISPARITY MAP IzqDer
	cl_mem DerIzq = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status); //CREAR BUFFER PARA DISPARITY MAP DerIzq

	cl_mem FinalMap = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status); //CREAR BUFFER PARA FINAL DISPARITY MAP 


	// ******** Read kernel file ********
	char *resize_kernel_file = leerFıchero("resize.cl");
	char *zncc_kernel_file = leerFıchero("zncc.cl");
	char *cross_check_kernel_file = leerFıchero("cross_check.cl");

	// ******* Init cl kernel from files *******
	imgDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
	imgDescriptor.image_width = w0;
	imgDescriptor.image_height = h0;
	imgDescriptor.image_depth = 8;
	imgDescriptor.image_row_pitch = w0 * resizear;

	dDisparity = (uint8_t*)malloc(new_width*new_height);
	Disparity = (uint8_t*)malloc(new_width*new_height);


	cl_kernel resize_kernel = file_to_kernel(ctx, resize_kernel_file, "resize");
	cl_kernel zncc_kernel = file_to_kernel(ctx, zncc_kernel_file, "zncc");
	cl_kernel cross_check_kernel = file_to_kernel(ctx, cross_check_kernel_file, "cross_check");

	// CREAR OBJETOS IMAGENES EN MEMROIA
	cl_mem clmemOrigImage0 = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, image0, &status);
	cl_mem clmemOrigImage1 = clCreateImage2D(ctx, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, image1, &status);


	// LLAMADA A LOS KERNELS
	status = 0;
	status = clSetKernelArg(resize_kernel, 0, sizeof(clmemOrigImage0), &clmemOrigImage0);
	status |= clSetKernelArg(resize_kernel, 1, sizeof(clmemOrigImage1), &clmemOrigImage1);
	status |= clSetKernelArg(resize_kernel, 2, sizeof(clmemImage0), &clmemImage0);
	status |= clSetKernelArg(resize_kernel, 3, sizeof(clmemImage1), &clmemImage1);
	status |= clSetKernelArg(resize_kernel, 4, sizeof(new_width), &new_width);
	status |= clSetKernelArg(resize_kernel, 5, sizeof(new_height), &new_height);

	status = clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);

	clFinish(queue);
	status = clEnqueueReadBuffer(queue, clmemImage0, CL_TRUE, 0, new_width*new_height, Disparity, 0, NULL, NULL);
	status = clEnqueueReadBuffer(queue, clmemImage1, CL_TRUE, 0, new_width*new_height, Disparity, 0, NULL, NULL);


	// MAPA DE DISPARIDAD DE IMAGEN 0 A IMAGEN 1
	status = 0;
	status = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImage0), &clmemImage0);
	status |= clSetKernelArg(zncc_kernel, 1, sizeof(clmemImage1), &clmemImage1);
	status |= clSetKernelArg(zncc_kernel, 2, sizeof(IzqDer), &IzqDer);
	status |= clSetKernelArg(zncc_kernel, 3, sizeof(new_width), &new_width);
	status |= clSetKernelArg(zncc_kernel, 4, sizeof(new_height), &new_height);
	status |= clSetKernelArg(zncc_kernel, 5, sizeof(half_winx), &half_winx);
	status |= clSetKernelArg(zncc_kernel, 6, sizeof(half_winy), &half_winy);
	status |= clSetKernelArg(zncc_kernel, 7, sizeof(win_area), &win_area);
	status |= clSetKernelArg(zncc_kernel, 8, sizeof(minDisp), &minDisp);
	status |= clSetKernelArg(zncc_kernel, 9, sizeof(maxDisp), &maxDisp);

	status = clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);

	// MAPA DE DISPARIDAD DE IMAGEN 1 A IMAGEN 0
	maxDisp *= -1;
	status = 0;
	status = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImage0), &clmemImage0);
	status |= clSetKernelArg(zncc_kernel, 1, sizeof(clmemImage1), &clmemImage1);
	status |= clSetKernelArg(zncc_kernel, 2, sizeof(DerIzq), &DerIzq);
	status |= clSetKernelArg(zncc_kernel, 3, sizeof(new_width), &new_width);
	status |= clSetKernelArg(zncc_kernel, 4, sizeof(new_height), &new_height);
	status |= clSetKernelArg(zncc_kernel, 5, sizeof(half_winx), &half_winx);
	status |= clSetKernelArg(zncc_kernel, 6, sizeof(half_winy), &half_winy);
	status |= clSetKernelArg(zncc_kernel, 7, sizeof(win_area), &win_area);
	status |= clSetKernelArg(zncc_kernel, 8, sizeof(minDisp), &minDisp);
	status |= clSetKernelArg(zncc_kernel, 9, sizeof(maxDisp), &maxDisp);

	status = clEnqueueNDRangeKernel(queue, zncc_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);


	// Cross checking kernel
	status = 0;
	status = clSetKernelArg(cross_check_kernel, 0, sizeof(IzqDer), &IzqDer);
	status |= clSetKernelArg(cross_check_kernel, 1, sizeof(DerIzq), &DerIzq);
	status |= clSetKernelArg(cross_check_kernel, 2, sizeof(FinalMap), &FinalMap);
	status |= clSetKernelArg(cross_check_kernel, 3, sizeof(threshold), &threshold);

	status = clEnqueueNDRangeKernel(queue, cross_check_kernel, 1, NULL, (const size_t*)&globalWorkSize1D, (const size_t*)&localWorkSize1D, 0, NULL, NULL);


	clFinish(queue);
	/*status = clEnqueueReadBuffer(queue, clmemDispMapCrossCheck, CL_TRUE, 0, Width*Height, dDisparity, 0, NULL, NULL);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "'cross_check_kernel': Failed to send the data to host !\n");
		abort();
	}
	*/

	// ******** run occlusion_filling & nomalize on host-code ********
	Disparity = SustituirCeros(dDisparity, new_width, new_height);
	normalizar(Disparity, new_width, new_height);


	if (frequencyEnd.QuadPart == 0)
		::QueryPerformanceFrequency(&frequencyEnd);
	::QueryPerformanceCounter(&endTime);


	printf("Tiempo total: %f s.\n", (endTime.QuadPart / double(frequencyEnd.QuadPart)) - (starTime.QuadPart / double(frequencyStart.QuadPart)));


	// ******** Save file to working directory (setup working directory may differ from IDEs) ********
	err = lodepng_encode_file("depthmap.png", Disparity, new_width, new_height, LCT_GREY, 8);
	free(image0);
	free(image1);
	free(resize_kernel_file);
	free(zncc_kernel_file);
	free(cross_check_kernel_file);
	free(dDisparity);
	free(Disparity);

	clReleaseKernel(resize_kernel);
	clReleaseKernel(zncc_kernel);
	clReleaseKernel(cross_check_kernel);
	clReleaseCommandQueue(queue);
	clReleaseContext(ctx);

	clReleaseMemObject(clmemOrigImage0);
	clReleaseMemObject(clmemOrigImage1);
	clReleaseMemObject(clmemImage0);
	clReleaseMemObject(clmemImage1);
	clReleaseMemObject(IzqDer);
	clReleaseMemObject(DerIzq);
	clReleaseMemObject(FinalMap);

	if (err) {
		printf("Error when saving the final 'depthmap.png' %u: %s\n", err, lodepng_error_text(err));
		return -1;
	}
	return 0;
}

/******************************************************************************
 *  Function that use to read kernel file
 */
char *leerFıchero(const char *filename)
{
	FILE *f = fopen(filename, "r");
	fseek(f, 0, SEEK_END);
	size_t size = ftell(f);
	fseek(f, 0, SEEK_SET);

	char *res = (char *)malloc(size + 1);
	if (!res) { // check error
		perror("Fail to read file, can not allocation memory !");
		abort();
	}

	// read file from stream
	if (fread(res, 1, size, f) < size) { // check error
		perror("Fail to read file, fread abort !");
		abort();
	}

	fclose(f);
	res[size] = '\0';
	return res;
}

/******************************************************************************
 *  Function that use to build kernel from file
 */
cl_kernel file_to_kernel(cl_context ctx, char const *kernel, char const *kernel_name)
{
	cl_int status;

	printf("Building kernel '%s'\n", kernel_name);
	// get kernel size
	size_t sizes[] = { strlen(kernel) };

	// create program
	cl_program program = clCreateProgramWithSource(ctx, 1, &kernel, sizes, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Fail to create program, in kernel file: '%s' !\n", kernel_name);
		abort();
	}

	// build program
	status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Fail to build program, in kernel file: '%s' !\n", kernel_name);
		abort();
	}

	// create our kernel from program
	cl_kernel res = clCreateKernel(program, kernel_name, &status);
	if (status != CL_SUCCESS) {
		fprintf(stderr, "Fail to create kernel, in kernel file: '%s' !\n", kernel_name);
		abort();
	}
	clReleaseProgram(program);
	printf("Succefully builed kernel '%s'\n", kernel_name);
	return res;
}

/******************************************************************************
 *  Replace each pixel with zero value with the nearest non-zero pixel value
 */
uint8_t* SustituirCeros(const uint8_t* dispMap, uint32_t w, uint32_t h) {
	int32_t i, j, ii, jj, k;
	bool flag; // flag for nearest non-zero pixel value

	uint8_t* result = (uint8_t*)malloc(w*h);

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			// If the value of the pixel is zero, perform the occlusion filling by nearest non-zero pixel value
			result[i*w + j] = dispMap[i*w + j];
			if (dispMap[i*w + j] == 0) {
				// Search of non-zero pixel in the neighborhood i,j, neighborhoodsize++
				flag = true;
				k = 0;
				while (flag) {
					k++;
					jj = -k;
					for (ii = -k; ii <= k && flag; ii++) {
						if (0 <= i + ii && i + ii < h && 0 <= j + jj && j + jj < w && dispMap[(i + ii)*w + (j + jj)] != 0) {
							result[i*w + j] = dispMap[(i + ii)*w + (j + jj)];
							flag = false;
							break;
						}
					}
					jj = k;
					for (ii = -k; ii <= k && flag; ii++) {
						if (0 <= i + ii && i + ii < h && 0 <= j + jj && j + jj < w && dispMap[(i + ii)*w + (j + jj)] != 0) {
							result[i*w + j] = dispMap[(i + ii)*w + (j + jj)];
							flag = false;
							break;
						}
					}
					ii = -k;
					for (jj = -k + 1; jj <= k - 1 && flag; jj++) {
						if (0 <= i + ii && i + ii < h && 0 <= j + jj && j + jj < w && dispMap[(i + ii)*w + (j + jj)] != 0) {
							result[i*w + j] = dispMap[(i + ii)*w + (j + jj)];
							flag = false;
							break;
						}
					}
					ii = k;
					for (jj = -k + 1; jj <= k && flag; jj++) {
						if (0 <= i + ii && i + ii < h && 0 <= j + jj && j + jj < w && dispMap[(i + ii)*w + (j + jj)] != 0) {
							result[i*w + j] = dispMap[(i + ii)*w + (j + jj)];
							flag = false;
							break;
						}
					}
				}
			}
		}
	}
	return result;
}

/******************************************************************************
 *  Normalize the final disparity map */

void normalizar(uint8_t* dispMap, uint32_t w, uint32_t h) {
	uint8_t maxValue = 0, minValue = UCHAR_MAX;
	uint32_t i;
	for (i = 0; i < w*h; i++) {
		if (dispMap[i] > maxValue) { maxValue = dispMap[i]; }
		if (dispMap[i] < minValue) { minValue = dispMap[i]; }
	}
	// Nomarlize to grey scale 0..255(UCHAR_MAX)
	maxValue -= minValue;
	for (i = 0; i < w*h; i++) {
		if(maxValue != 0) {
			dispMap[i] = (UCHAR_MAX*(dispMap[i] - minValue) / maxValue);
		}
		else {
			dispMap[i] = 15;
		}	
	}
}
