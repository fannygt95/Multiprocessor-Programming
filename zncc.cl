__kernel void zncc(__global uchar *img0, __global uchar *img1, __global uchar *DisMap, int w, int h, int window_w, int window_h, int win_area, int mind, int maxd) {

	int d;
	float average0, average1;     //Calcular medias
  	float desTipica0, desTipica1; //Calcular desviaciones típicas
	float aux0, aux1;	
  
	float correlation;
  	float biggestCorrelation; 
  	int ValorMatriz = maxd;
	double biggestCorrelation = -10000;

	const int i = get_global_id(0);
	const int j = get_global_id(1);

	for (d = mind; d <= maxd; d++) {
		average0 = average1 = 0;
		for (int win_y = -window_h; win_y < window_h; win_y++) {  // SUMAR LOS PIXELES DE LAS VENTANAS
			for (int win_x = -window_w; win_x < window_w; win_x++) {
				if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && 0 <= j + win_x - d && j + win_x - d < w) {
					average0 += img0[(i + win_y)*w + (j + win_x)];
					average1 += img1[(i + win_y)*w + (j + win_x - d)];
				}
			}
		}
		average0 = average0 / win_area;
		average1 = average1 / win_area;
		desTipica0 = desTipica1 = correlation = 0;

		for (int win_y = -window_h; win_y < window_h; win_y++) { // CALCULAR DESVIACIONES TIPICAS Y CORRELACION
			for (int win_x = -window_w; win_x < window_w; win_x++) {
				if (0 <= i + win_y && i + win_y < h && 0 <= j + win_x && j + win_x < w && 0 <= j + win_x - d && j + win_x - d < w) {
					aux0 = img0[(i + win_y)*w + (j + win_x)] - average0;
					aux1 = img1[(i + win_y)*w + (j + win_x - d)] - average1;
					correlation += aux0*aux1;
					desTipica0 += aux0*aux0;
					desTipica1 += aux1*aux1;
				}
			}
		}
		correlation = correlation / (native_sqrt(desTipica0)*native_sqrt(desTipica1));
		// WHICH IT IS THE BIGGEST DISPARITY
		if (correlation > biggestCorrelation) {
			biggestCorrelation = correlation;
			ValorMatriz = d;
		}
	}
	DisMap[i*w + j] = (uint)abs(ValorMatriz);
}
