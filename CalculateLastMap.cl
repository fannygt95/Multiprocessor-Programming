__kernel CalculateLastMap(__global uchar* firstMap, __global unchar* secondMap, __global uchar* lastMap, uint threshold) {

      constinti = get_global_id(0);
      
      if (abs((int)firstMap[i] - secondMap[i-firstMap[i] ]) > threshold)
          lastMap[i] = 0;
      }else{
          lastMap[i] = firstMap[i];
      }
