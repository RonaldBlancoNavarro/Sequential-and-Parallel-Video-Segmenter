#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <omp.h>

using namespace std;
using namespace cv;

vector<Mat> histogramas;
vector<Mat> frames;
vector<double> coeficientes;
vector<tuple<int, int>> tomas;
std::vector<double> gatheredVector;
double umbral = 0.6;
int contadorhistogramas = 0;
int totalFrames = 0;
Mat frameAuxVideo;
int numTrheads = 1;

void calcNormalizedHist(Mat &hsv, Mat &hist, int inicio, int j) // calculo de histogramas normalizados
{
    int h_bins = 50, s_bins = 60;
    int histSize[] = {h_bins, s_bins};
    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = {0, 180};
    float s_ranges[] = {0, 256};
    const float *ranges[] = {h_ranges, s_ranges};
    // Use the 0-th and 1-st channels
    int channels[] = {0, 1};

    calcHist(&hsv, 1, channels, Mat(), hist, 2, histSize, ranges, true, false);
    normalize(hist, hist, 0, 1, NORM_MINMAX, -1, Mat());

    // imshow("Display window", hist );//frames[contadorf]
    histogramas.push_back(hist);
}

void detectarTomas()
{
    int inicioToma = 0;
    for (int i = 0; i < coeficientes.size(); i++)
    {
        if (coeficientes[i] > umbral || i == coeficientes.size() - 1) // si el coeficiente es mayor al umbral, se considera toma
        {
            tomas.push_back(make_tuple(inicioToma, i)); // se guarda el inicio y fin de la toma
            if (i < coeficientes.size() - 1)
                inicioToma = i + 1;
        }
    }
}

void makeVideosTomas(int rank, int sizet, VideoCapture cap) // crear videos de las tomas
{

    int videosPorProceso = static_cast<int>(tomas.size()) / sizet; // cantidad de videos que le toca a cada proceso
    int ini = rank * videosPorProceso;
    int end = ini + videosPorProceso;
    #pragma omp parallel for
    for (int i = 1; i < tomas.size(); i++) // recorrer todas las tomas
    {    
        #pragma omp critical
        if (i >= ini && i < end) // Crear videos que le corresponden a cada proceso
        {
            string ruta_video = "/home/proyecto2/mpiopenmp/videos/videoo" + to_string(i) + ".avi";

            Size size = Size(frameAuxVideo.cols, frameAuxVideo.rows);
            VideoWriter video(ruta_video, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);

            int j = get<0>(tomas[i]);          // frame inicial de la toma
            cap.set(cv::CAP_PROP_POS_FRAMES, j);
            Mat frame;

            for (j; j <= get<1>(tomas[i]); j++) // recorrer los frames que le corresponden a cada toma y guardarlos en un video
            {
                if (cap.read(frame))
                {
                    video.write(frame);
                    frame.release();
                }
                else
                {
                    // cout << "Inicio" << inicio << "Fin" << fin << endl;
                    cout << "Rank: " << rank << " Error al leer el frame " << j << endl;
                    break;
                }
            }
            video.release();
        }
    }
}
// D:\Escritorio\proyecto1\paralelo\slipknot_1m_540p.mp4

void comparar(Mat &hist1, Mat &hist2) // comparacion de histogramas con el coeficiente de similitud de Bhattacharyya
{
    Mat hist1f;
    hist1.convertTo(hist1f, CV_32F);
    Mat hist2f;
    hist2.convertTo(hist2f, CV_32F);
    double bhattacharyya = compareHist(hist1f, hist2f, HISTCMP_BHATTACHARYYA);
    coeficientes.push_back(bhattacharyya);
    // cout << "Frame: " << contadorhistogramas << " Coeficiente Similitud Bhattacharyya con Siguiente histograma: " << bhattacharyya << endl;
    // contadorhistogramas++;
}

void compararHistogramas(int rank, int size, int fin)
{
    // Recorrer el vector de histogramas y compararlos
    if (histogramas.size() > 1) // Si hay mas de un histograma, compararlos
    {
        #pragma omp parallel for
        for (int i = 0; i < histogramas.size(); i++)
        {
            if (i < histogramas.size() - 1)
            {
                #pragma omp critical
                comparar(histogramas[i], histogramas[i + 1]); // Comparar el histograma actual con el siguiente
            }
        }
    }
    else
    {
        if (histogramas.size() == 0) // Si no hay histogramas, no se puede comparar
            cout << "No se encontraron histogramas" << endl;
        else // si solo hay un histograma, compararlo consigo mismo
            comparar(histogramas[0], histogramas[0]);
    }
}

void generarDocumento(double duration, int size)
{
    std::ofstream archivo("/home/proyecto2/mpiopenmp/paralelo_" + to_string(size) + ".txt", std::ios::trunc); // Abrir el archivo "paralelo.txt"
    // Comprobar si se ha abierto correctamente
    if (!archivo.is_open())
    {
        std::cerr << "Error al abrir el archivo paralelo.txt" << endl;
    }
    else
    {
        archivo << duration << endl; // Escribir el valor de la variable "duration" en el archivo
        archivo.close();
    }
}

void generarOutputFile(string file)
{
    int numFrames = 0;

    std::ofstream archivo(file + ".txt", std::ios::trunc); // Abrir el archivo file

    if (!archivo.is_open())
    {
        cout << "Error al abrir el archivo " << file << endl;
    }
    else
    {
        for (int i = 0; i < tomas.size(); i++)
        {
            numFrames += get<1>(tomas[i]) - get<0>(tomas[i]) + 1;
            archivo << "Toma: " << i << " Numero de frames: " << numFrames << endl; // Escribir el valor de las tomas y la cantidad de fremes en el archivo
            numFrames = 0;
        }
    }

    archivo.close();
}

void calcularUmbral()
{
    vector<double> coeficientesAUX = coeficientes;
    sort(coeficientesAUX.begin(), coeficientesAUX.end());

    double suma = 0;
    double contador = 0;

    for (int i = (static_cast<int>(coeficientesAUX.size()) / 10) * 9; i < coeficientesAUX.size(); i++)
    {
        if (coeficientes[i] >= 0.20)
        {
            suma += coeficientes[i];
            contador++;
        }
    }
    umbral = suma / contador; // calcular el umbral que es el promedio de los coeficientes de similitud de Bhattacharyya mas altos (25%)
    // cout << "Umbral: " << umbral << endl;
}

void generarDocCoeficientes(int rank)
{
    // crear un documento llamado como el numero de rank, con los coeficientes de similitud de Bhattacharyya de cada histograma
    std::ofstream archivo("/home/proyecto2/mpiopenmp/" + to_string(rank) + ".txt", std::ios::trunc); // Abrir el archivo "paralelo.txt"
    // Comprobar si se ha abierto correctamente
    if (!archivo.is_open())
    {
        std::cerr << "Error al abrir el archivo " << to_string(rank) << ".txt" << endl;
    }
    else
    {
        for (int i = 0; i < coeficientes.size(); i++)
        {
            archivo << coeficientes[i] << endl; // Escribir el valor de la variable "duration" en el archivo
        }
        archivo.close();
    }
}

void leerDocsCoeficientes(int rank, int size)
{
    // leer los documentos llamados como el numero de rank, con los coeficientes de similitud de Bhattacharyya de cada histograma y agragar esto a el vector coeficientes
    int i = 0;

    if (rank == 0)
    {
        i = 1;
    }
    else
    {
        coeficientes.clear();
        i = 0;
    }

    for (i; i < size; i++) // cargar los coeficientes de los otros procesos
    {
        std::ifstream archivo("/home/proyecto2/mpiopenmp/" + to_string(i) + ".txt");
        if (!archivo.is_open())
        {
            std::cerr << "Error al abrir el archivo " << to_string(i) << ".txt" << endl;
        }
        else
        {
            // leer el archivo los coeficientes separados por salto de linea y agregar los coeficientes al vector coeficientes
            std::string line;
            while (std::getline(archivo, line))
            {
                coeficientes.push_back(std::stod(line));
            }
            archivo.close();
        }
    }
}

int main(int argc, char **argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_timeO = MPI_Wtime(); // inicia el tiempo
    double start_time = MPI_Wtime();  // inicia el tiempo
    
    //int numThreads = 2; // Número de hilos deseado
    //omp_set_num_threads(numThreads);

    if (argv[2] == NULL)
    {
        cout << "No se ha especificado el nombre del archivo de salida" << endl;
        return -1;
    }

    if (rank == 0)
    {
        cout << endl;
        cout << "Algoritmo Segmentacion Paralela de Video" << endl;
    }

    string nombreArchivo = argv[1];
    string outputFile = argv[2];
    VideoCapture cap("/home/proyecto2/mpiopenmp/" + nombreArchivo);

    string numHilos = argv[3];
    // comprobar que se ingreso correctamente el numero de procesos
    if (numHilos == "2" || numHilos == "3" ||  numHilos == "4" || numHilos == "6" || numHilos == "8")
    {
        numTrheads = atoi(argv[2]); // obtener la cantidad de procesos que se van a utilizar
        omp_set_num_threads(numTrheads);
    }
    else
    {
        if (rank == 0)
            std::cerr << "Error al ingresar el numero de Trheads" << std::endl;

        return -1;
    }

    int cont = 0;

    // obtener la cantidad de frames del video
    totalFrames = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT));

    // calcular la cantidad de frames que le corresponden a cada proceso
    int framesPorProceso = totalFrames / size;
    int inicio = rank * framesPorProceso;
    int fin = inicio + framesPorProceso;

    // Abrir el archivo de video
    if (!cap.isOpened())
    {
        if (rank == 0)
            std::cerr << "Error al abrir el archivo de video, especifique correctamente su nombre y extencion" << std::endl;
        return -1;
    }

    Mat frame, resizedFrame, hsvFrame, hist;
    string ruta_frame;
    int frameIndex = inicio;
    cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    cv::Size targetSize(256, 144); // Tamaño objetivo mas pequeño
    #pragma omp for
    for (frameIndex = inicio; frameIndex < fin; ++frameIndex)
    { // cada rank almacena los frames que le corresponden
        if (cap.read(frame))
        {
            if (frameIndex == inicio)
            { // cargar el primer frame de cada proceso para el video de tomas
                frameAuxVideo = frame;
            }
            // frames.push_back(frame);
            // ruta_frame = "C:/home/proyecto1/paralelo/frames/"  + to_string(frameIndex) + ".jpg";
            // imwrite(ruta_frame, frame);

            cv::resize(frame, resizedFrame, targetSize);
            cv::cvtColor(resizedFrame, hsvFrame, cv::COLOR_BGR2HSV);
            //      // Calcular el histograma normalizado de la imagen en espacio de color HSV
            calcNormalizedHist(hsvFrame, hist, inicio, frameIndex);
            frame.release();
            hsvFrame.release();
            resizedFrame.release();
            hist.release();
        }
    }

    if (rank == 0)
    {
        MPI_Barrier(MPI_COMM_WORLD); // Esperar a que todos los procesos terminen
        double end_time = MPI_Wtime();
        double duration = end_time - start_time;
        cout << endl
             << "Tiempo de ejecucion calcNormalizedHist: " << duration << " segundos." << endl
             << endl;
    }
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    start_time = MPI_Wtime();
    compararHistogramas(rank, size, fin);
    generarDocCoeficientes(rank);
    if (rank == 0)
    {
        MPI_Barrier(MPI_COMM_WORLD); // Esperar a que todos los procesos terminen
        double end_time = MPI_Wtime();
        double duration = end_time - start_time;
        cout << endl
             << "Tiempo de ejecucion compararHistogramas: " << duration << " segundos." << endl
             << endl;
    }
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    leerDocsCoeficientes(rank, size);

    // if (rank != 0)
    // {
    //     // imprimir coeficientes
    //     for (int i = 0; i < coeficientes.size(); i++)
    //     {
    //         cout << "Frame: " << i << " Coeficiente Similitud Bhattacharyya con Siguiente histograma: " << coeficientes[i] << endl;
    //     }

    //     cout<<endl<<coeficientes.size()<<endl<<endl;
    // }
    calcularUmbral();
    detectarTomas();

    makeVideosTomas(rank, size, cap);

    destroyAllWindows();
    cap.release();
    
    if (rank == 0)
    {
        MPI_Barrier(MPI_COMM_WORLD); // Esperar a que todos los procesos terminen

        double end_time = MPI_Wtime();
        double duration = end_time - start_timeO;
        cout << endl
             << "Tiempo de ejecucion total: " << duration << " segundos." << endl
             << endl;
        generarOutputFile(outputFile);
        generarDocumento(duration, size);
    }
    else
    {
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}