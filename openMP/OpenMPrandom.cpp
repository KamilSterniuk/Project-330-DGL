#include <iostream>
#include <vector>
#include <chrono>  // Dodaj chrono do pomiaru czasu
#include <iomanip> // Dodaj iomanip do formatowania liczb
#include <omp.h>   // Dodaj OpenMP
#include <cstdlib> // Dodaj rand()

// Struktura CSR
struct CSRMatrix
{
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<double> values;
    int rows;
    int cols;
};

// Funkcja do wykonywania SpMM z wykorzystaniem OpenMP
std::vector<std::vector<double>> spmm(const CSRMatrix &A, const std::vector<std::vector<double>> &B)
{
    // Sprawdzenie wymiarów
    if (A.cols != B.size())
    {
        throw std::invalid_argument("Dimensions of matrices are not compatible for multiplication.");
    }

    int rows = A.rows;
    int cols = B[0].size();
    std::vector<std::vector<double>> C(rows, std::vector<double>(cols, 0.0));

    // Mnożenie równoległe
#pragma omp parallel for
    for (int i = 0; i < rows; ++i)
    {
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
        {
            int a_col = A.col_idx[j];
            double a_val = A.values[j];

            for (int k = 0; k < cols; ++k)
            {
#pragma omp atomic
                C[i][k] += a_val * B[a_col][k];
            }
        }
    }

    return C;
}

// Funkcja do wyświetlania wyniku w formacie pełnej macierzy
void printDense(const std::vector<std::vector<double>> &C)
{
    std::cout << "Dense matrix:" << std::endl;
    std::cout << std::fixed << std::setprecision(0); // Ustawienia formatowania
    for (const auto &row : C)
    {
        for (double val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

// Funkcja do generowania macierzy CSR o rozmiarze 100x100
CSRMatrix generateRandomCSRMatrix(int size)
{
    CSRMatrix mat;
    mat.rows = mat.cols = size;

    // Generowanie wierszy z losowymi wartościami w formacie CSR
    mat.row_ptr.push_back(0); // Początek pierwszego wiersza
    for (int i = 0; i < size; ++i)
    {
        int row_length = rand() % (size / 2); // Losowa liczba niezerowych elementów w wierszu
        for (int j = 0; j < row_length; ++j)
        {
            mat.col_idx.push_back(rand() % size);     // Losowy indeks kolumny
            mat.values.push_back((rand() % 100) + 1); // Losowa wartość
        }
        mat.row_ptr.push_back(mat.row_ptr.back() + row_length); // Aktualizacja wskaźnika końca wiersza
    }

    return mat;
}

int main()
{
    int size = 100; // Rozmiar macierzy 100x100

    // Ustawienie liczby wątków na 4
    omp_set_num_threads(4);

    // Generowanie losowych macierzy CSR i B
    CSRMatrix A = generateRandomCSRMatrix(size);

    // Tworzymy gęstą macierz B o wymiarach 100x100
    std::vector<std::vector<double>> B(size, std::vector<double>(size, 1.0)); // Gęsta macierz 100x100 z wartościami 1

    // Pomiar czasu wykonania SpMM
    auto start = std::chrono::high_resolution_clock::now(); // Start czasu

    // Mnożenie A * B
    std::vector<std::vector<double>> C = spmm(A, B);

    auto end = std::chrono::high_resolution_clock::now(); // Koniec czasu

    // Obliczenie różnicy czasu
    std::chrono::duration<double> elapsed = end - start;
    std::cout << std::fixed << std::setprecision(6); // Czas z 6 miejscami po przecinku
    std::cout << "Czas wykonania SpMM: " << elapsed.count() << " sekund" << std::endl;

    // Wyświetlanie wyniku w formacie pełnej macierzy
    // std::cout << "Matrix C in dense format:" << std::endl;
    // printDense(C);

    return 0;
}
