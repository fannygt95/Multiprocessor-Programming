#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdbool.h>
#include <time.h>
#include "lodepng.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>


const int resizear= 4;    // valor por el que dividimos la imagen
const uint32_t half_winx= 8;    
const uint32_t half_winy= 15;   
const int threshold= 8;    // Threshold para calcular el mapa final
const uint32_t win_area= 527;  // (half_winx x 2 + 1) x (half_winY x 2 + 1) = win_area

int maxDisp= 64;   
int minDisp= 0;

typedef struct IMAGEN {
	cl_mem_object_type image_type;
	size_t image_width;
	size_t image_height;
	size_t image_depth;
	size_t image_size;
	size_t image_row_pitch;
	size_t image_slice_pitch;
	cl_uint num_mip_levels;
	cl_uint num_samples;
	cl_mem buffer;
} IMAGEN;

IMAGEN imgDescriptor;        // Image descriptor contains type and dimensions of the image
cl_image_format format = {CL_RGBA, CL_UNSIGNED_INT8};


char *leerFichero(const char *filename);
cl_kernel file_to_kernel(cl_context ctx, char const *kernel, char const *kernel_name);
void normalizar(uint8_t* lastMap, uint32_t width, uint32_t height);
uint8_t* SustituirCeros(const uint8_t* dispMap, uint32_t w, uint32_t h);


int32_t main()
{
    uint8_t *image0, *image1; // Imagenes iniciales
    uint8_t *dDisparity, *Disparity;

    uint32_t err;                       // Error code, 0 is OK
    uint32_t new_width, new_height;     // nuevo tamaño
    uint32_t wL, hL, wR, hR;            // tamaños originales

    struct timespec totalStartTime, totalEndTime;

    cl_context ctx;
    cl_command_queue queue;
    cl_int status;

    const size_t localWorkSize[] = {16, 16};       
    const size_t globalWorkSize[] = {504, 736};   

    const size_t localWorkSize1D[]  = {localWorkSize[0]*localWorkSize[1]};      
    const size_t globalWorkSize1D[] = {globalWorkSize[0]*globalWorkSize[1]};    


    // CARGAR IMAGENES EN MEMORIA Y COMPROBAR POSIBLES ERRORES
    lodepng_decode32_file(&image0, &wL, &hL, "im0.png");
    lodepng_decode32_file(&image1, &wR, &hR, "im1.png");
    if(wL!=wR || hL!=hR) {
        printf("Error, tamaños diferentes.\n");
        free(image0);
        free(image1);
        return -1;
    }
    new_width= wL/resizear;
    new_height= hL/resizear;

    clock_gettime(CLOCK_MONOTONIC, &totalStartTime);

    // KERNEL
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = {CL_CONTEXT_PLATFORM, 0, 0};

    status = clGetPlatformIDs(1, &platform, NULL);
    int gpu = 1; // O : CPU, 1 : GPU
    status = clGetDeviceIDs(platform, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device, NULL);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext( props, 1, &device, NULL, NULL, &status );
    queue = clCreateCommandQueue( ctx, device, 0, &status );
    
    // MEMORIA PARA IMAGENES Y DISPARITY MAPS
    cl_mem clmemImage0 = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status);
    cl_mem clmemImageR = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status);
    
    cl_mem IzqDer = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status);
    cl_mem DerIzq = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status);
    
    cl_mem FinalMap = clCreateBuffer(ctx, CL_MEM_READ_ONLY, new_width*new_height, 0, &status);
    
    char *resize_kernel_file       = read_kernel_file("ReduceGrayMatrix.cl");
    char *zncc_kernel_file         = read_kernel_file("ZNCC.cl");
    char *cross_check_kernel_file  = read_kernel_file("CalculateLastMap.cl");

    imgDescriptor.image_type = CL_MEM_OBJECT_IMAGE2D;
    imgDescriptor.image_width = wL;
    imgDescriptor.image_height = hL;
    imgDescriptor.image_depth = 8;
    imgDescriptor.image_row_pitch = wL * resizear;

    dDisparity = (uint8_t*) malloc(new_width*new_height);
    Disparity  = (uint8_t*) malloc(new_width*new_height);


    cl_kernel resize_kernel     = build_kernel_from_file(ctx, resize_kernel_file, "ReduceGrayMatrix");
    cl_kernel zncc_kernel       = build_kernel_from_file(ctx, zncc_kernel_file, "ZNCC");
    cl_kernel cross_check_kernel= build_kernel_from_file(ctx, cross_check_kernel_file, "CalculateLastMap");

    // CREAR OBJETOS IMAGENES EN MEMROIA******** Create images memory objects ********
    cl_mem clmemimage0 = clCreateImage2D(ctx, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, image0, &status);
    cl_mem clmemimage1 = clCreateImage2D(ctx, CL_MEM_READ_ONLY|CL_MEM_USE_HOST_PTR, &format, imgDescriptor.image_width, imgDescriptor.image_height, imgDescriptor.image_row_pitch, image1, &status);
    
    // LLAMADA A LOS KERNELS
    status = 0;
    status  = clSetKernelArg(resize_kernel, 0, sizeof(clmemimage0), &clmemimage0);
    status |= clSetKernelArg(resize_kernel, 1, sizeof(clmemimage1), &clmemimage1);
    status |= clSetKernelArg(resize_kernel, 2, sizeof(clmemImage0), &clmemImage0);
    status |= clSetKernelArg(resize_kernel, 3, sizeof(clmemImage1), &clmemImage1);
    status |= clSetKernelArg(resize_kernel, 4, sizeof(new_width), &new_width);
    status |= clSetKernelArg(resize_kernel, 5, sizeof(new_height), &new_height);

    status = clEnqueueNDRangeKernel(queue, resize_kernel, 2, NULL, (const size_t*)&globalWorkSize, (const size_t*)&localWorkSize, 0, NULL, NULL);

    clFinish(queue);
    status = clEnqueueReadBuffer(queue, clmemImage0, CL_TRUE,  0, new_width*new_height, Disparity, 0, NULL, NULL);
    status = clEnqueueReadBuffer(queue, clmemImage1, CL_TRUE,  0, new_width*new_height, Disparity, 0, NULL, NULL);
   
    // MAPA DE DISPARIDAD DE IMAGEN 0 A IMAGEN 1
    status = 0;
    status  = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImage0), &clmemImage0);
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
    status  = clSetKernelArg(zncc_kernel, 0, sizeof(clmemImage0), &clmemImage0);
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
    status  = clSetKernelArg(cross_check_kernel, 0, sizeof(IzqDer), &IzqDer);
    status |= clSetKernelArg(cross_check_kernel, 1, sizeof(DerIzq), &DerIzq);
    status |= clSetKernelArg(cross_check_kernel, 2, sizeof(FinalMap), &FinalMap);
    status |= clSetKernelArg(cross_check_kernel, 3, sizeof(THRESHOLD), &THRESHOLD);
    
    status = clEnqueueNDRangeKernel(queue, cross_check_kernel, 1, NULL, (const size_t*)&globalWorkSize1D, (const size_t*)&localWorkSize1D, 0, NULL, NULL);
   
    clFinish(queue);
    status = clEnqueueReadBuffer(queue, clmemDispMapCrossCheck, CL_TRUE, 0, new_width*new_height, dDisparity, 0, NULL, NULL);
    
    // ******** run occlusion_filling & nomalize on host-code ********
    Disparity = SustituirCeros(dDisparity, new_width, new_height);
    normalizar(Disparity, new_width, new_height);

    clock_gettime(CLOCK_MONOTONIC, &totalEndTime); // Ending time
    printf("Tiempo total del ZNCC: %f s. ***\n", (double)(totalEndTime.tv_sec - totalStartTime.tv_sec) + (double)(totalEndTime.tv_nsec - totalStartTime.tv_nsec)/1000000000);


    // GUARDAR MAPA FINAL Y LIBERAR ESPACIO EN MEMORIA
    lodepng_encode_file("depthmap.png", Disparity, new_width, new_height, LCT_GREY, 8);
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

    clReleaseMemObject(clmemimage0);
    clReleaseMemObject(clmemimage1);
    clReleaseMemObject(clmemImage0);
    clReleaseMemObject(clmemImage1);
    clReleaseMemObject(IzqDer);
    clReleaseMemObject(DerIzq);
    clReleaseMemObject(LastMap);

}

char *leerFichero(const char *filename){ //LEER KERNEL

    FILE *fich = fopen(filename, "r");
    fseek(fich, 0, SEEK_END);
    size_t size = ftell(fich);
    fseek(fich, 0, SEEK_SET);

    char *res = (char *) malloc(size+1);

    if(fread(res, 1, size, fich) < size) { // check error
        perror("Error de lectura");
        abort();
    }

    fclose(fich);
    res[size] = '\0';
    return res;
}

/******************************************************************************
 *  Function that use to build kernel from file
 */
cl_kernel file_to_kernel(cl_context ctx, char const *kernel, char const *kernel_name){ //CONSTRUIR KERNEL
    cl_int status;

    size_t sizes[] = { strlen(kernel) };

    cl_program program = clCreateProgramWithSource(ctx, 1, &kernel, sizes, &status);

    status = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    cl_kernel res = clCreateKernel(program, kernel_name, &status);

    clReleaseProgram(program);
    return res;
}

/******************************************************************************
*  Replace each pixel with zero value with the nearest non-zero pixel value
*/
uint8_t* SustituirCeros(const uint8_t* dispMap, uint32_t w, uint32_t h){
	int32_t i, j;
	int32_t auxValor;
	int32_t win_y, win_x;
	bool aux; 

	uint8_t* res = (uint8_t*)malloc(w*h);

	for (i = 0; i < h; i++) {
		for (j = 0; j < w; j++) {
			result[i*w + j] = dispMap[i*w + j];
			if (dispMap[i*w + j] == 0) {
				aux = true;
				auxValor = 0;
				while (aux) {
					auxValor++;
					win_x = -auxValor;
					for (win_y = -auxValor; win_y <= auxValor && aux; win_y++) {
						if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && dispMap[(i + win_y)*w + (j + win_x)] != 0) {
							result[i*w + j] = dispMap[(i + win_y)*w + (j + win_x)];
							aux = false;
							break;
						}
					}
					win_x = auxValor;
					for (win_y = -auxValor; win_y <= auxValor && aux; win_y++) {
						if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && dispMap[(i + win_y)*w + (j + win_x)] != 0) {
							result[i*w + j] = dispMap[(i + win_y)*w + (j + win_x)];
							aux = false;
							break;
						}
					}
					win_y = -auxValor;
					for (win_x = -auxValor + 1; win_x <= auxValor - 1 && aux; win_x++) {
						if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && dispMap[(i + win_y)*w + (j + win_x)] != 0) {
							result[i*w + j] = dispMap[(i + win_y)*w + (j + win_x)];
							aux = false;
							break;
						}
					}
					win_y = auxValor;
					for (win_x = -auxValor + 1; win_x <= auxValor && aux; win_x++) {
						if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && dispMap[(i + win_y)*w + (j + win_x)] != 0) {
							result[i*w + j] = dispMap[(i + win_y)*w + (j + win_x)];
							aux = false;
							break;
						}
					}
				}
			}
		}
	}
	return res;
}

/******************************************************************************
 *  Normalize the final disparity map
 */
void normalizar(uint8_t* lastMap, uint32_t width, uint32_t height) {
	uint8_t maxValue = 0;
	uint8_t minValue = UCHAR_MAX;
	uint32_t i;
	for (i = 0; i < width*height; i++) {
		if(lastMap [i]>maxValue){
			maxValue=lastMap[i];
		}
		
        	if(lastMap [i]<minValue){
			minValue=lastMap[i];
		}
	}	
	maxValue -= minValue;
	for (i = 0; i < width*height; i++) {
		lastMap [i] = (UCHAR_MAX*(lastMap i] - minValue)/maxValue);
	}
}
