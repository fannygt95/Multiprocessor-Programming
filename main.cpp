#include "lodepng.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

/*int const width = 2940;
int const hight = 2016;
*/
int const disparity = 260;
int windowSize = 9;

int main(int argc, char *argv[]) {
	
	const char* filename0 = argc > 1 ? argv[1] : "im0.png";
	//load and decode
	std::vector<unsigned char> image0;
	unsigned width, height;
	unsigned error0 = lodepng::decode(image0, width, height, filename0);
	//if there's an error, display it
	if (error0) std::cout << "decoder error " << error0 << ": " << lodepng_error_text(error0) << std::endl;
	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...

	unsigned im0[2940][2016];
	int x = 0;
	for (int i = 0; i < height; i++){
		for (int y = 0; y < width; y++){
			im0[i][y] = image0[x];
			x++;
		}
	}

	const char* filename1 = argc > 1 ? argv[1] : "im1.png";
	//load and decode
	std::vector<unsigned char> image1;
	//unsigned width, height;
	unsigned error1 = lodepng::decode(image1, width, height, filename1);
	//if there's an error, display it
	if (error1) std::cout << "decoder error " << error1 << ": " << lodepng_error_text(error1) << std::endl;
	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...
	
	

	
	unsigned average;
	for (unsigned i = windowSize / 2; i < height - windowSize / 2; i++){
		for (int j = windowSize / 2; j < width - windowSize / 2; j++){
			for (int d = 0; d < disparity; d++){
				for (int win_y = j - windowSize / 2; win_y < j + windowSize / 2 ; win_y++){
					for (int win_x = i - windowSize / 2; win_x <  i + windowSize / 2; win_x++){
						
					}
				}

			}
		}
	}


	

	system("pause");
}
