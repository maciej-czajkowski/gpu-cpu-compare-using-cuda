

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdbool>
#include <vector>
#include <time.h>
#include <thread>
#include <math.h>

#define MAX_THREADS (1200)
#define CUDA_CORES (1024)

typedef double result;

enum Operation{
    AVG,
    STD_DEV
};


typedef struct {
    float open;
    float close;
    float high;
    float low;
} quota;


typedef struct {
    float avg;
    float dev;
} results;


std::vector<std::string> splitCsv(const std::string s, const std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}


void saveToFile(const std::string &filename, const result *res, const size_t length) {
    std::ofstream out(filename);
    for (size_t i = 0; i < length; i++) {
        out << res[i] << std::endl; 
    }
    out.close();
}


void calculateAvg(const quota &in, result &out) {
    out = (in.close + in.high + in.low + in.open) / 4;
}


__global__ void cudaAvg(const quota* in, result* out, const size_t N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int i = threadIdx.x;
    if (i < N) {
        out[i] = (in[i].close + in[i].high + in[i].low + in[i].open) / 4;
    }
}


void calculateStdDev(const quota &in, result &out) {
    double avg = (in.close + in.high + in.low + in.open) / 4;
    out = sqrt((pow(in.close - avg, 2) + pow(in.high - avg, 2) + pow(in.low - avg, 2) + pow(in.open - avg, 2)) / 4);
}


__global__ void cudaStdDev(const quota *in, result *out, const size_t N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    //int i = threadIdx.x;
    if (i < N) {
        out[i] = sqrt((pow(in[i].close - ((in[i].close + in[i].high + in[i].low + in[i].open) / 4), 2) +
            pow(in[i].high - ((in[i].close + in[i].high + in[i].low + in[i].open) / 4), 2) +
            pow(in[i].low - ((in[i].close + in[i].high + in[i].low + in[i].open) / 4), 2) +
            pow(in[i].open - ((in[i].close + in[i].high + in[i].low + in[i].open) / 4), 2)) / 4);
    }
}


void runThreadsCPU(Operation op, const quota *quotasArray, result *resultsArray, const size_t length, double &time) {
    std::string operation = op == AVG ? "average" : "standard deviation";
    std::cout << "########## " << "CPU" << " ###########" << std::endl;
    std::cout << "Calculating " << operation << " using CPU." << std::endl;
    
    auto start = std::chrono::system_clock::now();

    const float cycles = length / MAX_THREADS;  // number of threads per run
    std::thread* threadsArray = new std::thread[length];

    for (size_t i = 0; i < cycles + 1; i++) {
        for (size_t j = 0; j < MAX_THREADS; j++) {
            size_t currentPt = (i * MAX_THREADS) + j;
            try {
                // cout << (i*MAX_THREADS)+j << endl;
                if (currentPt < length) {
                    switch (op)
                    {
                    case AVG:
                        threadsArray[currentPt] = std::thread(calculateAvg, std::ref(quotasArray[currentPt]), std::ref(resultsArray[currentPt]));
                        break;
                    case STD_DEV:
                        threadsArray[currentPt] = std::thread(calculateStdDev, std::ref(quotasArray[currentPt]), std::ref(resultsArray[currentPt]));
                        break;
                    default:
                        break;
                    }
                }
                else {
                   // cout << "Finished on " << currentPt << endl;
                    break;
                }
            }
            catch (const std::exception& e) {
               std::cout << currentPt << std::endl;
               std::cerr << e.what() << std::endl;
            }
        }
        for (size_t j = 0; j < MAX_THREADS; j++) {
            size_t currentPt = (i * MAX_THREADS) + j;
            if (currentPt < length && threadsArray[currentPt].joinable()) {
                threadsArray[currentPt].join();
            }
            else {
                break;
            }
        }
    }
        
    delete[] threadsArray;
   
    auto stop = std::chrono::system_clock::now();

    std::cout << "Done." << std::endl;

    std::chrono::duration<double> elapsedTime = stop - start;
    time = elapsedTime.count();
    std::cout << "duration = " << time << "s" << std::endl << std::endl;
}

cudaError_t runThreadsGPU(Operation op, const quota *quotasArray, result *resultsArray, const size_t length, double &calculateTime, double &transferTime) {
    cudaError_t cudaStatus;
    quota *dev_quotasArray = NULL;
    result *dev_resultsArray = NULL;

    int blockSize;   
    int minGridSize;
    int gridSize;

    std::string operation = op == AVG ? "average" : "standard deviation";
    std::cout << "########## " << "GPU" << " ###########" << std::endl << std::endl;
    std::cout << std::endl << "Calculating " << operation << " using GPU." << std::endl;
    std::cout << "Memory transfer to device... " << std::endl;

    auto start = std::chrono::system_clock::now();

    /* only one device */
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaSetDevice failed!" << std::endl;
        return cudaStatus;
    }

    /* allocating memory */
    cudaStatus = cudaMalloc((void**)&dev_quotasArray, length * sizeof(quota));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for quotas array!" << std::endl;
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((void**)&dev_resultsArray, length * sizeof(result));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMalloc failed for results array!" << std::endl;
        return cudaStatus;
    }


    /* copy to buffers */
    cudaStatus = cudaMemcpy(dev_quotasArray, quotasArray, length * sizeof(quota), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for quotas array!" << std::endl;
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_resultsArray, resultsArray, length * sizeof(result), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for results array!" << std::endl;
        return cudaStatus;
    }

    auto stop = std::chrono::system_clock::now();

    std::cout << "Memory transfer to device done." << std::endl;

    std::chrono::duration<double> elapsedTime = stop - start;

    transferTime = elapsedTime.count();

    std::cout << "duration = " << transferTime << "s" << std::endl << std::endl << std::endl;

    std::cout << "Calculating..." << std::endl;
    start = std::chrono::system_clock::now();

    
    int blocks = ceil(length / CUDA_CORES) + 1;

    /* run kernels */
    switch (op)
    {
    case AVG:
        cudaAvg <<<blocks, CUDA_CORES >>> (dev_quotasArray, dev_resultsArray, length);
        break;
    case STD_DEV:
        cudaStdDev <<<blocks, CUDA_CORES >>> (dev_quotasArray, dev_resultsArray, length);
        break;
    default:
        break;
    }

    /* check for errors */
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cuda kernel start failed!" << std::endl;
        return cudaStatus;
    }

    /* wait for kernels */
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceSynchronize failed!" << std::endl;
        return cudaStatus;
    }

    stop = std::chrono::system_clock::now();

    std::cout << "Calculation done." << std::endl;

    elapsedTime = stop - start;

    calculateTime = elapsedTime.count();
    std::cout << "duration = " << calculateTime << "s" << std::endl << std::endl;

    std::cout << "Memory transfer to host... " << std::endl;
    start = std::chrono::system_clock::now();

    /* get results from GPU*/
    cudaStatus = cudaMemcpy(resultsArray, dev_resultsArray, length * sizeof(result), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaMemcpy to host failed!" << std::endl;
        return cudaStatus;
    }

    cudaFree(dev_resultsArray);
    cudaFree(dev_quotasArray);

    /* reset device */
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed!" << std::endl;
        return cudaStatus;
    }

    stop = std::chrono::system_clock::now();

    std::cout << "Memory transfer to host done." << std::endl;

    elapsedTime = stop - start;
    std::cout << "duration = " << elapsedTime.count() << "s" << std::endl << std::endl << std::endl;

    transferTime += elapsedTime.count();


    std::cout << "Total duration = " << calculateTime + transferTime << "s" << std::endl << std::endl;


    return cudaStatus;
}


int main()
{
    std::cout << "Bit Coin App" << std::endl;

    const std::string MY_NAN = "NaN";
    const std::string DELIMETER = ",";

    std::ifstream infile("C:\\Users\\marty\\Downloads\\bitcoin.csv");
    if (!infile.good()) {
        std::cerr << "Error: Couldn't open file" << std::endl;
        return 4;
    }

    std::cout << "Parsing csv file." << std::endl;


    std::string entry;
    // skip 1st line
    std::getline(infile, entry);
    std::vector<quota> quotas;
    long int counter = 0;
    while (std::getline(infile, entry)) {
        if (entry.find(MY_NAN) == std::string::npos) {
            // cout << counter << endl;
            counter++;
            std::vector<std::string> entrySplited = splitCsv(entry, DELIMETER);
            quota q;
            q.open = std::stof(entrySplited[1]);
            q.high = std::stof(entrySplited[2]);
            q.low = std::stof(entrySplited[3]);
            q.close = std::stof(entrySplited[4]);
            quotas.push_back(q);
        }
    }
    std::cout << std::endl;

    std::cout << "Starting preprocessing." << std::endl;

    size_t entryCount = quotas.size();
    std::cout << "entries count = " << entryCount << ", parsing done." << std::endl;
    quota* quotasArray = new quota[entryCount];

    
    for (size_t i = 0; i < entryCount; i++) {
        /* cpy vector to array for performance*/
        quotasArray[i] = quotas[i];
    }


    std::cout << "Preprocesing finished." << std::endl;

    result* resultsArray = new result[entryCount];
    memset(resultsArray, 0, entryCount);

    double CPUtimes[2] = { 0 };
    double GPUCalculatetimes[2] = { 0 };
    double GPUTransferTime[2] = { 0 };

    /* ################### CPU ###################*/
    
    runThreadsCPU(AVG, quotasArray, resultsArray, entryCount, CPUtimes[0]);
    saveToFile("CPU_AVG.txt", resultsArray, entryCount);
    memset(resultsArray, 0, entryCount);

    runThreadsCPU(STD_DEV, quotasArray, resultsArray, entryCount, CPUtimes[1]);
    saveToFile("CPU_STDDEV.txt", resultsArray, entryCount);
    memset(resultsArray, 0, entryCount);
    


    /* ################### GPU ###################*/
    
    cudaError_t cudaStatus = cudaSuccess;

    for (int i = 0; i < 2; i++) {
        std::cout << std::endl << "Run no." << i + 1 << std::endl;
        cudaStatus = runThreadsGPU(AVG, quotasArray, resultsArray, entryCount, GPUCalculatetimes[0], GPUTransferTime[0]);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Cuda failed!" << std::endl;
            return 1;
        }
        memset(resultsArray, 0, entryCount);
    }

    std::cout << std::endl << "Run no.3" << std::endl;
    cudaStatus = runThreadsGPU(AVG, quotasArray, resultsArray, entryCount, GPUCalculatetimes[0], GPUTransferTime[0]);
    saveToFile("GPU_AVG.txt", resultsArray, entryCount);
    memset(resultsArray, 0, entryCount);

    for (int i = 0; i < 2; i++) {
        std::cout << std::endl << "Run no." << i + 1 << std::endl;
        runThreadsGPU(STD_DEV, quotasArray, resultsArray, entryCount, GPUCalculatetimes[1], GPUTransferTime[1]);
        if (cudaStatus != cudaSuccess) {
            std::cerr << "Cuda failed!" << std::endl;
            return 1;
        }
        memset(resultsArray, 0, entryCount);
    }

    std::cout << std::endl << "Run no.3" << std::endl;
    runThreadsGPU(STD_DEV, quotasArray, resultsArray, entryCount, GPUCalculatetimes[1], GPUTransferTime[1]);
    saveToFile("GPU_STDDEV.txt", resultsArray, entryCount);
    memset(resultsArray, 0, entryCount);



    delete[] resultsArray;
    delete[] quotasArray;


    std::cout << std::endl << std::endl << "##########  Summary  ##########" << std::endl;
    std::cout << "Including transfer time:" << std::endl;
    std::cout << "Average calculation time CPU/GPU, GPU is " << CPUtimes[0] / (GPUCalculatetimes[0] + GPUTransferTime[0]) 
        << "% times faster." << std::endl;

    std::cout << "Standard deviation calculation time CPU/GPU, GPU is " << CPUtimes[1] / (GPUCalculatetimes[1] + GPUTransferTime[1]) 
        << "% times faster." << std::endl << std::endl;

    std::cout << "Excluding transfer time:" << std::endl;
        std::cout << "Average calculation time CPU/GPU, GPU is " << CPUtimes[0] / (GPUCalculatetimes[0])
        << "% times faster." << std::endl;

    std::cout << "Standard deviation calculation time CPU/GPU, GPU is " << CPUtimes[1] / (GPUCalculatetimes[1])
        << "% times faster." << std::endl << std::endl;

    return 0;
}
