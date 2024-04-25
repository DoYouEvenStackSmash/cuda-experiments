#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "image_handler.h"

#include <cmath>

// int binloader()
int main() {
    // Open the binary file for reading
    std::ifstream file("frames.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return 1;
    }

    // Determine the size of the file
    file.seekg(0, std::ios::end);
    std::streampos fileSize = file.tellg();
    std::cout << fileSize << std::endl;
    file.seekg(0, std::ios::beg);

    // Calculate the size of each array
    std::streampos arraySize = fileSize / 600;

    // Read the file into 600 arrays
    std::vector<std::vector<char>> arrays(600);
    
    for (int i = 0; i < 600; ++i) {
        // Resize the vector to hold the data for one array
        std::cout << arraySize << std::endl;
        arrays[i].resize(arraySize);
        // Read data into the vector
        file.read(arrays[i].data(), arraySize);
        // Check for errors
        if (file.bad()) {
            std::cerr << "Error reading file." << std::endl;
            return 1;
        }
    }

    // Close the file
    file.close();

    // Access the data in the arrays as needed
    // Example: Print the first 10 bytes of each array
    // for (int i = 0; i < 600; ++i) {
    //     std::cout << "Array " << i << ": ";
    //     for (int j = 0; j < 400; ++j) {
    //         std::cout << static_cast<int>(arrays[i][j]) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
