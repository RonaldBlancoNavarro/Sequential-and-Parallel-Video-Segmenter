#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <tuple>
#include <fstream>

using namespace std;
using namespace cv;

vector<Mat> histogramas;
vector<Mat> frames;
vector<double> coeficientes;
vector<tuple<int, int>> tomas;
double umbral = 0.6;
int contadorhistogramas = 0;

void calcNormalizedHist(Mat &hsv, Mat &hist) // calculo de histograma normalizado
{
    // numero de bins
    int histSize = 256;

    // dar rangos (B,G,R)
    float h_range[] = {0, 180}; // rango para el canal H
    float s_range[] = {0, 256}; // rango para el canal S
    float v_range[] = {0, 256}; // rango para el canal V
    const float *histRange[] = {h_range, s_range, v_range};

    // Establecer parámetro de histograma
    bool uniform = true, accumulate = false;

    // Separar los canales de la imagen
    vector<Mat> hsv_planes;
    split(hsv, hsv_planes);

    // Calcular el histograma para cada canal
    Mat h_hist, s_hist, v_hist;
    calcHist(&hsv_planes[0], 1, 0, Mat(), h_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&hsv_planes[1], 1, 0, Mat(), s_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&hsv_planes[2], 1, 0, Mat(), v_hist, 1, &histSize, histRange, uniform, accumulate);

    // Dibujar los histogramas para B, G y R
    int hist_w = 512, hist_h = 400;
    int bin_w = cvRound((double)hist_w / histSize);
    Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

    // Normalizar el histograma respecto a la cantidad de pixeles de la imagen
    normalize(h_hist, h_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(s_hist, s_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
    normalize(v_hist, v_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

    // Dibujar el histograma en una imagen
    for (int i = 1; i < histSize; i++)
    {
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(h_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(h_hist.at<float>(i))),
             Scalar(255, 0, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(s_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(s_hist.at<float>(i))),
             Scalar(0, 255, 0), 2, 8, 0);
        line(histImage, Point(bin_w * (i - 1), hist_h - cvRound(v_hist.at<float>(i - 1))),
             Point(bin_w * (i), hist_h - cvRound(v_hist.at<float>(i))),
             Scalar(0, 0, 255), 2, 8, 0);
    }

    histogramas.push_back(histImage);
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
        coeficientes[i] = NULL;
    }
    for (int i = 0; i < tomas.size(); i++)
    {
        cout << "Toma: " << i << ": " << get<0>(tomas[i]) << " - " << get<1>(tomas[i]) << endl;
    }
}

void makeVideosTomas() // crear videos de las tomas detectadas
{
    histogramas.clear();
    coeficientes.clear();

    int x = 1;
    for (int i = 0; i < tomas.size(); i++)
    {
        string ruta_video = "/home/proyecto1/secuencial/videos/videoo" + to_string(x) + ".avi"; 
        Mat frame = frames[0]; // asumiendo que tu vector de frames se llama 'frames'
        Size size = Size(frame.cols, frame.rows);
        VideoWriter video(ruta_video, VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, size, true);

        for (int j = get<0>(tomas[i]); j <= get<1>(tomas[i]); j++)
        {
            video.write(frames[j]);
            frames[j]=NULL;
        }
        x++;
        video.release();
    }

    frames.clear();
    tomas.clear();
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
            histogramas[i] = NULL;
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
    std::ofstream archivo("secuencial.txt", std::ios::trunc); // Abrir el archivo "secuencial.txt"
    // Comprobar si se ha abierto correctamente
    if (!archivo.is_open())
    {
        std::cerr << "Error al abrir el archivo secuencial.txt" << endl;
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
        if(coeficientes[i] >= 0.50){
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

    cout << endl;
    cout << "Algoritmo Segmentacion Secuencial de Video" << endl;

    string nombreArchivo = argv[1];

    // Abrir el archivo de video
    VideoCapture cap(nombreArchivo);
    if (!cap.isOpened())
    {
        std::cerr << "Error al abrir el archivo de video, especifique correctamente su nombre y extencion" << std::endl;
        return -1;
    }

    // Iterar por cada cuadro del video
    Mat frame, hsvFrame, hist;
    while (cap.read(frame))
    {
        Mat *FFrame = new Mat();
        frame.copyTo(*FFrame);
        frames.push_back(*FFrame); // guardar el frame

        // Convertir la imagen a espacio de color HSV
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

        // Asegurarse de que los valores de ambas capas estén en el rango de 0 a 255
        Mat channels[3];
        split(hsvFrame, channels);
        channels[0] *= 0.5;           // La capa de tono se divide por 2 para que esté en el rango de 0 a 127
        channels[1] *= 255.0 / 360.0; // La capa de saturación se multiplica por 255/360 para que esté en el rango de 0 a 255
        channels[2] *= 255.0 / 255.0; // La capa de valor se multiplica por 255/255 para que esté en el rango de 0 a 255
        merge(channels, 3, hsvFrame);

        // Calcular el histograma normalizado de la imagen en espacio de color HSV
        calcNormalizedHist(hsvFrame, hist);

        // Presionar  ESC  para salir
        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }
    compararHistogramas();
    calcularUmbral();
    detectarTomas();
    makeVideosTomas();
    destroyAllWindows();

    cap.release();

    clock_t end = clock();
    double duration = double(end - start) / CLOCKS_PER_SEC;
    cout << "Tiempo de ejecucion: " << duration << " segundos" << endl;

    generarDocumento(duration);
    return 0;
}
