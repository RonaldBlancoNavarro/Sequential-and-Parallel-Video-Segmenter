#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>

using namespace std;
using namespace cv;

Mat combined_histogram;
vector<Mat> histogramas;
vector<Mat> frames;
vector<double> coeficientes;
vector<tuple<int, int>> tomas;
double umbral = 0.6;
int contadorhistogramas = 0;
int totalFrames = 0; 
int numTrheads = 1; 

void calcNormalizedHist(Mat &hsv, Mat &hist) // calculo de histograma normalizado
{
    int h_bins = 50, s_bins = 60;
    int histSize[] = { h_bins, s_bins };
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };

    calcHist( &hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false );
    normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );

    //imshow("Display window", hist );//frames[contadorf] 
     histogramas.push_back(hist);
}


void comparar(Mat &hist1, Mat &hist2) // comparacion de 2 histogramas
{
    Mat hist1f;
    hist1.convertTo(hist1f, CV_32F);
    Mat hist2f;
    hist2.convertTo(hist2f, CV_32F);
    // calculo de coeficiente de similitud
    double bhattacharyya = compareHist(hist1f, hist2f, HISTCMP_BHATTACHARYYA);
    coeficientes.push_back(bhattacharyya);
    // cout << "Frame: " << contadorhistogramas << " Coeficiente Similitud Bhattacharyya con Siguiente histograma: " << bhattacharyya << endl;
    contadorhistogramas++;
}

void detectarTomas() // determinar el inicio y fin de las tomas segun el umbral de corte
{
    int inicioToma = 0;
    for (int i = 0; i < coeficientes.size(); i++)
    {
        if (coeficientes[i] > umbral || i == coeficientes.size() - 1) // si el coeficiente es mayor al umbral, se considera toma
        {
            tomas.push_back(make_tuple(inicioToma, i));
            if (i < coeficientes.size() - 1)
                inicioToma = i + 1;
        }
        //coeficientes[i] = NULL;
    }
    for (int i = 1; i < tomas.size(); i++)
    {
        cout << "Toma: " << i << ": " << get<0>(tomas[i]) << " - " << get<1>(tomas[i]) << endl;
    }
}

void makeVideosTomas() // crear videos de las tomas detectadas
{
    int x = 1;
    #pragma omp for
    for (int i = 1; i < tomas.size(); i++)
    {
        string ruta_video = "/home/proyecto2/openmp/videos/videoo" + to_string(x) + ".avi";
        Mat frame = frames[0]; // asumiendo que tu vector de frames se llama 'frames'
        Size size = Size(frame.cols, frame.rows);
        VideoWriter video(ruta_video, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);

        for (int j = get<0>(tomas[i]); j <= get<1>(tomas[i]); j++)
        {
            video.write(frames[j]);
            //frames[j]=NULL;
        }
        x++;
        video.release();
    }
}

void compararHistogramas() // Recorrer el vector de histogramas y compararlos
{
    if (histogramas.size() > 1)
    {
        for (int i = 0; i < histogramas.size(); i++)
        {
            if (i < histogramas.size() - 1)
            {
                comparar(histogramas[i], histogramas[i + 1]); // Comparar el histograma actual con el siguiente
            }
            // clear the content of histogramas[i] without changing the size of the vector
            //histogramas[i] = NULL;
        }
    }
    else
    {
        if (histogramas.size() == 0)
            cout << "No se encontraron histogramas" << endl;
        else
            comparar(histogramas[0], histogramas[0]);
    }
}

void generarDocumento(double duration)
{
    std::ofstream archivo("/home/proyecto2/openmp/openmp.txt", std::ios::trunc); // Abrir el archivo "openmp.txt"
    // Comprobar si se ha abierto correctamente
    if (!archivo.is_open())
    {
        std::cerr << "Error al abrir el archivo openmp.txt" << endl;
    }
    else
    {
        archivo << duration << endl; // Escribir el valor de la variable "duration" en el archivo
        archivo.close();
    }
}

void calcularUmbral()
{
    vector<double> coeficientesAUX = coeficientes;
    sort(coeficientesAUX.begin(), coeficientesAUX.end());

    double suma = 0;
    double contador = 0;
    int coeficientesAUXSize = static_cast<int>(coeficientesAUX.size());

    for (int i = (coeficientesAUXSize/10)*9; i < coeficientesAUX.size(); i++)
    {
        if(coeficientes[i] >= 0.20){
        suma += coeficientes[i];
        contador++;
        }
    }
    umbral = suma / contador; // calcular el umbral que es el promedio de los coeficientes de similitud de Bhattacharyya mas altos (25%)
    cout << "Umbral: " << umbral << endl;
}

int main(int argc, char **argv)
{
    clock_t start = clock(); // Obtiene el tiempo actual
    clock_t start1 = clock(); // Obtiene el tiempo actual

    cout << endl;
    cout << "Algoritmo Segmentacion Secuencial de Video" << endl;

    string nombreArchivo = argv[1];

    // Abrir el archivo de video
    VideoCapture cap("/home/proyecto2/openmp/" + nombreArchivo);
    if (!cap.isOpened())
    {
        std::cerr << "Error al abrir el archivo de video, especifique correctamente su nombre y extencion" << std::endl;
        return -1;
    }

    string numHilos = argv[2];
    // comprobar que se ingreso correctamente el numero de procesos
    if (numHilos == "2" || numHilos == "3" ||  numHilos == "4" || numHilos == "6" || numHilos == "8")
    {
        numTrheads = atoi(argv[2]); // obtener la cantidad de procesos que se van a utilizar
        omp_set_num_threads(numTrheads);
    }
    else
    {
        std::cerr << "Error al ingresar el numero de Trheads" << std::endl;
        return -1;
    }

    totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));
    // Iterar por cada cuadro del video
    Mat frame, hsvFrame, hist,resizedFrame;
    cv::Size targetSize(256, 144); // Tamaño objetivo mas pequeño
    
    #pragma omp for
    for(int i=0; i<totalFrames; i++){
        if (cap.read(frame))
        {
            frames.push_back(frame);

            cv::resize(frame, resizedFrame, targetSize);
            cv::cvtColor(resizedFrame, hsvFrame, cv::COLOR_BGR2HSV);
    //      // Calcular el histograma normalizado de la imagen en espacio de color HSV
            calcNormalizedHist(hsvFrame, hist);
            frame.release();
            hsvFrame.release();
            resizedFrame.release();
            hist.release();
        }
    }


    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    cout << endl<< "Tiempo de ejecucion calcNormalizedHist: " << duration << " segundos" << endl << endl;

    start = clock(); // Obtiene el tiempo actual
    compararHistogramas();
    end = clock();
    duration = double(end - start) / CLOCKS_PER_SEC;
    cout<< endl << "Tiempo de ejecucion compararHistogramas: " << duration << " segundos" << endl << endl;

    start = clock(); // Obtiene el tiempo actual
    calcularUmbral();
    end = clock();
    duration = double(end - start) / CLOCKS_PER_SEC;
    cout<< endl << "Tiempo de ejecucion calcularUmbral: " << duration << " segundos" << endl << endl;

    start = clock(); // Obtiene el tiempo actual
    detectarTomas();
    end = clock();
    duration = double(end - start) / CLOCKS_PER_SEC;
    cout<< endl << "Tiempo de ejecucion detectarTomas: " << duration << " segundos" << endl << endl;

    start = clock(); // Obtiene el tiempo actual
    makeVideosTomas();
    end = clock();
    duration = double(end - start) / CLOCKS_PER_SEC;
    cout<< endl << "Tiempo de ejecucion makeVideosTomas: " << duration << " segundos" << endl << endl;

    destroyAllWindows();

    //  system("pause");
    cap.release();

     end = clock();
     duration = double(end - start1) / CLOCKS_PER_SEC;
    cout<< endl << "Tiempo de ejecucion total: " << duration << " segundos" << endl << endl;

    generarDocumento(duration);
    return 0;
}
