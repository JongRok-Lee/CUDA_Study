#include <iostream>

int main(int argc, char* argv[], char* envp[]) {
    std::cout << "Number of arguments: " << argc << std::endl;
    for (int i = 0; i < argc; i++) {
        std::cout << "Argument " << i << ": " << argv[i] << std::endl;
    }

    std::cout << "Environment variables:" << std::endl;
    for (int i = 0; envp[i] != nullptr; i++) {
        std::cout << "variable " << i << ": " << envp[i] << std::endl;
    }
    return 0;
}