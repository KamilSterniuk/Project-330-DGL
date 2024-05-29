#include <torch/torch.h>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

int main() {
    std::string path = "./"; // Ścieżka do plików z minibatchami

    for (const auto &entry : fs::directory_iterator(path)) {
        if (entry.path().extension() == ".pt") {
            torch::Tensor batch;
            torch::load(batch, entry.path().string());
            std::cout << "Wczytano minibatch z pliku: " << entry.path() << std::endl;
            std::cout << "Rozmiar tensora: " << batch.sizes() << std::endl;
        }
    }

    return 0;
}
