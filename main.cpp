#include "lodepng.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

unsigned width, height;
unsigned average = 0;
int const disparity = 260;
int windowSize = 9;
double correlation = 0;
unsigned biggestCorrelation = 0;


unsigned char reduceim0[735][504];
unsigned char reduceim1[735][504];


void reduceTheMatrix(vector<unsigned char> imagen, unsigned char reducematrix[735][504]);
void project(unsigned char im0[735][504], unsigned char im1[735][504]);
unsigned lodepng_encode_file(const char* filename, const unsigned char* image, unsigned w, unsigned h, LodePNGColorType colortype, unsigned bitdepth);

int main(int argc, char *argv[]){
	
	////////////////////////////////// GETING THE PIXELS FROM THE FIRST IMAGE ///////////////////////////////////////////////
	const char* filename0 = argc > 1 ? argv[1] : "im0.png";
	//load and decode
	std::vector<unsigned char> image0;
	unsigned error0 = lodepng::decode(image0, width, height, filename0);
	//if there's an error, display it
	if (error0) std::cout << "decoder error " << error0 << ": " << lodepng_error_text(error0) << std::endl;
	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...


	///////////////////////////////// GETING THE PIXELS FROM THE SECOND IMAGE ///////////////////////////////////////////////
	const char* filename1 = argc > 1 ? argv[1] : "im1.png";
	//load and decode
	std::vector<unsigned char> image1;
	unsigned error1 = lodepng::decode(image1, width, height, filename1);
	//if there's an error, display it
	if (error1) std::cout << "decoder error " << error1 << ": " << lodepng_error_text(error1) << std::endl;
	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...


	reduceTheMatrix(image0, reduceim0);
	reduceTheMatrix(image1, reduceim1);
	project(reduceim0, reduceim1);
}


void reduceTheMatrix(vector<unsigned char> imagen, unsigned char reducematrix[735][504]){
	
	width = 735; height = 504;
	int R = 0, G = 1, B = 2;
	int a = 0, b;
	for (int i = 0; i < height ; i++){
		b = 0;
		for (int x = 0; x < width; x++){
			reducematrix[a][b] = imagen[R] * 0.2126 + imagen[G] * 0.7152 + imagen[B] * 0.0722;
			b++;
			R = R + 16;
			G = G + 16;
			B = B + 16;
		}
		a++;
	}
}


void project(unsigned char im0[735][504], unsigned char im1[735][504]){
	double average0, average1;
	for (unsigned i = windowSize / 2; i < height - windowSize / 2; i++){
		for (int j = windowSize / 2; j < width - windowSize / 2; j++){
			
			for (int d = 0; d < disparity; d++){
				int count0 = 0, count1 = 0;
				
				for (int win_y = j - windowSize / 2; win_y < windowSize; win_y++){
					for (int win_x = i - windowSize / 2; win_x < windowSize; win_x++){
						count0 = count0 + im0[win_y][win_x]; //SUMA DE TODAS LAS VENTANAS
						count1 = count1 + im1[win_y][win_x];
					}
				}
				average0 = count0 / (windowSize * windowSize); //MEDIA DE LAS VENTANAS
				average1 = count1 / (windowSize * windowSize);

				double covarianza = 0;
				double desTipica0 = 0, desTipica1 = 0;
				
				
				for (int win_y = j - windowSize / 2; win_y < windowSize; win_y++){
					for (int win_x = i - windowSize / 2; win_x < windowSize; win_x++){
				//////////////////////////////////  ZNCC VALUE  ////////////////////////////////////////
						double aux0 = im0[win_y][win_x];
						double aux1 = im1[win_y][win_x];
						// covarianza//
						covarianza = covarianza + ((aux0 - average0) * (aux1 - average1));
						
						// desviacion tipica imagen 0//						
						aux0 = aux0 - average0;
						desTipica0 = desTipica0 + pow(aux0, 2);						
						
						// desviacion tipica imagen 1//						
						aux1 = aux1 - average1;
						desTipica1 = desTipica1 + pow(aux1, 2);
						

					}
				}
				desTipica0 = sqrt(desTipica0 / (windowSize * windowSize));
				desTipica1 = sqrt(desTipica1 / (windowSize * windowSize));

				covarianza = covarianza / (windowSize * windowSize);
				correlation = covarianza / (desTipica0 * desTipica1);
				// WE ARE GOING TO COMPARING ONE WINDOW IN THE IMG0 WITH 260 WINDOWS IN THE IMG1 AND WE TAKE THE BIGGEST ONE 
				// WHICH IT IS THE BIGGEST DISPARITY
				if (correlation > biggestCorrelation){ 
					biggestCorrelation = correlation;
					int newPixelY = i; //WE SAVE THE COORDINATES OF THE PIXEL WITH THE BIGGEST CORRELATION
					int newPixelX = j;

				}
			}
			
		}
	}
	
}

	


	

