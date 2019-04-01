#include "lodepng.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream> 
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;

unsigned width, height;
unsigned average = 0;
int const disparity = 65;
const int windowSize = 9;
double correlation = 0;
unsigned char biggestCorrelation;


vector<vector<unsigned char>> reduceim0;
vector<vector<unsigned char>> reduceim1;

// MAPAS DE DISPARIDAD //
vector<vector<unsigned char>> IzqDer;
vector<vector<unsigned char>> DerIzq;

// MAPA FINAL//
vector<vector<unsigned char>> FinalMap;

// MAPA FINAL EN UN VECTOR
vector<unsigned char>FinalVector;


void getTheImagenInAVector(const char* filename, vector<unsigned char> &image);
void ReduceGrayMatrix(vector<unsigned char> imagen, vector<vector<unsigned char>>&reducematrix);
void ZNCC(vector<vector<unsigned char>> im0, vector<vector<unsigned char>>im1, vector<vector<unsigned char>>&DisMap);
unsigned char operations(int j, int i, double vector0[windowSize*windowSize], vector<vector<unsigned char>> im1, int average0, double desTipica0);
void MapToZero(unsigned char DispMap[504][735]);
void CalculateLastMap(vector<vector<unsigned char>>firstMap, vector<vector<unsigned char>>secondMap, vector<vector<unsigned char>>&lastMap);
void SustituirCeros(vector<vector<unsigned char>>&lastMap);
void MapToVector(vector<vector<unsigned char>>Map, vector<unsigned char>&vector);

int main(int argc, char *argv[]){

	vector<unsigned char> image0;
	vector<unsigned char> image1;

	vector<unsigned char> aux;

	const char* filename0 = argc > 1 ? argv[1] : "im0.png";
	const char* filename1 = argc > 1 ? argv[1] : "im1.png";

	const char* save_left = "grayscale_left.png";
	const char* save_right = "dgrayscale_right.png";
/*
	vector<unsigned char>cristinita(40);
	double amaia[5][8];
	double marco = 2.5;
	double natalia = 0;
	for (int alvaro = 0; alvaro < 5; alvaro++){
		for (int ixaka = 0; ixaka < 8; ixaka++){
			amaia[alvaro][ixaka] = marco + natalia;
			cout << amaia[alvaro][ixaka] << " ";
			natalia++;
		}
		cout << "\n";
	}
	int raquel = 0;
	for (int paula = 0; paula < 4; paula++){
		for (int rosa = 0; rosa < 4; rosa++){
			cristinita[raquel] = amaia[paula][rosa];
			raquel++;
		}
	}
	encodeOneStep(".\\images\\out.png", cristinita, 5, 8);
*/


	getTheImagenInAVector(filename0, image0);
	//getTheImagenInAVector(filename1, image1);

	ReduceGrayMatrix(image0, reduceim0);

	MapToVector(reduceim0, aux);

	
	lodepng::encode("DisparityMap.png", aux, width, height, LCT_GREY, 8);

	//ReduceGrayMatrix(image1, reduceim1);
	/*
	image0.clear(); image0.shrink_to_fit();
	image1.clear(); image1.shrink_to_fit();

	//MapToZero(IzqDer);
	//MapToZero(DerIzq);
	ZNCC(reduceim0, reduceim1, IzqDer);
	ZNCC(reduceim1, reduceim0, DerIzq);

	reduceim0.clear(); reduceim0.shrink_to_fit();
	reduceim1.clear(); reduceim1.shrink_to_fit();

	CalculateLastMap(IzqDer, DerIzq, FinalMap);
	SustituirCeros(FinalMap);
	MapToVector(FinalMap, FinalVector);

	//IMPRIMIR FOTOGRAFIA	
	lodepng::encode(".\\DisparityMap.png", FinalVector, width, height);
	*/
}

void getTheImagenInAVector(const char* filename, vector<unsigned char> &image){

	unsigned error0 = lodepng::decode(image, width, height, filename);
	//if there's an error, display it
	if (error0) std::cout << "decoder error " << error0 << ": " << lodepng_error_text(error0) << std::endl;
	//the pixels are now in the vector "image", 4 bytes per pixel, ordered RGBARGBA..., use it as texture, draw it, ...


}

void ReduceGrayMatrix(vector<unsigned char> imagen, vector<vector<unsigned char>>&reduceimagen){
	vector <unsigned char> auxvector;
	unsigned char aux;
	vector<vector<unsigned char>> auxImagen;
	int R = 0, G = 1, B = 2;

	for (int i = 0; i < height; i++){
		for (int x = 0; x < width; x++){
			aux = imagen[R] * 0.2126 + imagen[G] * 0.7152 + imagen[B] * 0.0722;
			if (x + i * width < imagen.size()){
				auxvector.push_back(aux);
			}
			R = R + 4;
			G = G + 4;
			B = B + 4;
		}
		auxImagen.push_back(auxvector);
		auxvector.erase(auxvector.begin(), auxvector.end());
	}

	for (int i = 0; i < height; i+=4){
		for (int x = 0; x < width; x+=4){
			auxvector.push_back(auxImagen[i][x]);
		}
		reduceimagen.push_back(auxvector);
		auxvector.erase(auxvector.begin(), auxvector.end());
	}
	width = width / 4; height = height / 4;	
}

void ZNCC(vector<vector<unsigned char>> im0, vector<vector<unsigned char>> im1, vector<vector<unsigned char>>&DisMap){
	double count0 = 0, count1 = 0;  //Contador para la suma de los datos de las ventanas
	double vector0[windowSize * windowSize]; //Guarda los datos de la ventana de la primera imagen
	double vector1[windowSize * windowSize]; //Guarda los datos de la ventana de la segunda imagen
	unsigned average0, average1; //Calcular medias
	double desTipica0 = 0, desTipica1 = 0; //Calcular desviaciones típicas
	double covarianza = 0; // Calcular covarianza

	unsigned aux;
	vector<unsigned char> vectorAux;

	for (int i = windowSize / 2; i < height + 1 - windowSize / 2; i++){  //RECORRER IMAGEN
		for (int j = windowSize / 2; j < width + 1 - windowSize / 2; j++){

			int m = 0;
			count0 = 0;
			average = 0;
			for (int y = i - windowSize / 2; y < i + 1 + windowSize / 2; y++) {// RECORRER VENTANA IMAGEN 0
				for (int x = j - windowSize / 2; x < j + 1 + windowSize / 2; x++) {
				
					count0 = count0 + im0[y][x]; //Sumatorio datos ventana imagen 0
					vector0[m] = im0[y][x]; //Guardar datos
					m++;
				}
			}
			average0 = count0 / (pow(windowSize,2)); //Media ventana imagen0
			int aux0 = 0;	
			desTipica0 = 0;
			for (m = 0; m < sizeof(vector0) / sizeof(*vector0); m++){//RECORRER OTRA VEZ VENTANA IMAGEN 0, esta vez usamos el vector
				aux0 = vector0[m] - average0;
				desTipica0 = desTipica0 + (pow(aux0, 2));
			}
			desTipica0 = sqrt(desTipica0 / (pow(windowSize, 2)));

			
			aux = operations(i, j, vector0, im1, average0, desTipica0);
			vectorAux.push_back(aux);
			//cout << IzqDer[i][j] << " ";
			}
		DisMap.push_back(vectorAux);
		vectorAux.erase(vectorAux.begin(), vectorAux.end());
	}

}

unsigned char operations(int hei, int wid, double vector0[windowSize*windowSize], vector<vector<unsigned char>> im1, int average0, double desTipica0){

	int vector1[windowSize * windowSize];
	int h = wid;
	int g = hei;
	int countDisparity = 0;
	biggestCorrelation = 0;
	while (g < height - windowSize + 1/ 2 && countDisparity < disparity){
		while (h < width - windowSize + 1/ 2 && countDisparity < disparity){
			int count1 = 0;
			int n = 0;
			for (int win_y = g - windowSize / 2; win_y < g + 1 + windowSize / 2; win_y++){ //CONTADOR DE LOS DATOS VENTANA IMAGEN 1
				for (int win_x = h - windowSize / 2; win_x < h + 1 + windowSize / 2; win_x++){
					count1 = count1 + im1[win_y][win_x];
					vector1[n] = im1[win_y][win_x]; //CREAR VECTOR DE DATOS VENTANA IMAGEN 1
					n++;
				}
			}
			int average1 = 0;
			average1 = count1 / (pow(windowSize, 2));
			int aux1 = 0;
			double desTipica1 = 0;
			for (n = 0; n < sizeof(vector1) / sizeof(*vector1); n++){//RECORRER OTRA VEZ VENTANA IMAGEN 1, esta vez usamos el vector
				aux1 = vector1[n] - average1;
				desTipica1 = desTipica1 + (pow(aux1, 2));
			}
			desTipica1 = sqrt(desTipica1 / (pow(windowSize, 2)));

			double auxcovarianza = 0;
			double covarianza = 0;
			for (n = 0; n < sizeof(vector1) / sizeof(*vector1); n++) { //CALCULAR COVARIANZA CON LOS DATOS GUARDADOS EN LOS VECTORES
				auxcovarianza = (vector0[n] - average0) * (vector1[n] - average1);
				covarianza = covarianza + auxcovarianza;
			}
			covarianza = covarianza / (pow(windowSize, 2));

			correlation = covarianza / (desTipica0 * desTipica1); // CORRELACIÓN
			// WE ARE GOING TO COMPARING ONE WINDOW IN THE IMG0 WITH 260 WINDOWS IN THE IMG1 AND WE TAKE THE BIGGEST ONE 
			// WHICH IT IS THE BIGGEST DISPARITY
			
			if (correlation > biggestCorrelation){
				biggestCorrelation = correlation;				
			}
			countDisparity++;
			h++;
		}
		g++;
	}
	return biggestCorrelation;
}

/*void MapToZero(double DispMap[735][504]) {
	for (int i = 0; i < 735; i++) {
		for (int j = 0; j < 504; j++) {
			DispMap[i][j] = 0;
		}
	}
}
*/

void CalculateLastMap(vector<vector<unsigned char>> firstMap, vector<vector<unsigned char>> secondMap, vector<vector<unsigned char>>&lastMap){
	
	int resta = 0;
	for (int i = 4; i < 504 + 1 + windowSize / 2; i++) {
		for (int j = 4; j < 735 + 1 + windowSize / 2; j++) {
			cout << "1. " << IzqDer[i][j] << " 2. " << DerIzq[i][j] << "\n";
			cout << "1. " << firstMap[i][j] << " 2. " << secondMap[i][j]  << "\n";

			resta = firstMap[i][j] - secondMap[i][j];
			resta = abs(resta);
			if (resta > 8){
				lastMap[i][j] = 0;
			}
			else{
				lastMap[i][j] = resta;
			}
		}
	}
}

void SustituirCeros(vector<vector<unsigned char>>&lastMap){
	int aux = 0;
	for (int i = windowSize / 2; i < height + 1 - windowSize / 2; i++){
		for (int j = windowSize / 2; j < width + 1 - windowSize / 2; j++){
			if (lastMap[i][j] == 0){
				aux = lastMap[i][j + 1] + lastMap[i][j - 1] + lastMap[i + 1][j] + lastMap[i - 1][j] + lastMap[i - 1][j - 1] + lastMap[i - 1][j + 1] + lastMap[i + 1][j - 1] + lastMap[i + 1][j + 1];
				aux = aux / 8;
				lastMap[i][j] = aux;
				//cout << lastMap[j][i] << " ";

			}
		}
	}
}

void MapToVector(vector<vector<unsigned char>> Map, vector<unsigned char> &vector){
	
	//int n = 0;
	//char *vector = (char*)malloc(sizeof (char));
	for (int i = 0; i < height; i++){
		for (int j = 0; j < width; j++){
			vector.push_back(Map[i][j]); 
			//cout << vector[n] << " ";
			//n++;
		}
	}
}

