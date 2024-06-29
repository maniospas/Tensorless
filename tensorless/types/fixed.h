/*
Copyright 2024 Emmanouil Krasanakis

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef FIXED_H
#define FIXED_H


#include <iostream>
#include <stdexcept>
#include <random>

template <typename T, std::size_t N>
class Fixed {
public:
    Fixed() {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] = T(0);
    }

    Fixed(const Fixed<T, N> &other) {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] = other[i]; 
    }

    Fixed<T, N>& set(std::size_t index, T value) {
        if (index >= N) 
            throw std::out_of_range("Index out of range");
        data[index] = value;
        return *this;
    }

    T& operator[](std::size_t index) {
        if (index >= N) 
            throw std::out_of_range("Index out of range");
        return data[index];
    }

    const T& operator[](std::size_t index) const {
        if (index >= N) 
            throw std::out_of_range("Index out of range");
        return data[index];
    }

    Fixed<T, N> operator+(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (std::size_t i = 0; i < N; ++i) 
            result[i] = data[i] + other[i];
        return result;
    }

    Fixed<T, N> operator-(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (std::size_t i = 0; i < N; ++i) 
            result[i] = data[i] - other[i];
        return result;
    }

    Fixed<T, N> operator*(const Fixed<T, N>& other) const {
        Fixed<T, N> result;
        for (std::size_t i = 0; i < N; ++i) 
            result[i] = data[i] * other[i];
        return result;
    }

    Fixed<T, N> operator*(const T &other) const {
        Fixed<T, N> result;
        for (std::size_t i = 0; i < N; ++i) 
            result[i] = data[i] * other;
        return result;
    }

    Fixed<T, N>& operator+=(const Fixed<T, N>& other) {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] += other[i];
        return *this;
    }

    Fixed<T, N>& operator-=(const Fixed<T, N>& other) {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] -= other[i];
        return *this;
    }

    Fixed<T, N>& operator*=(const Fixed<T, N>& other) {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] *= other[i];
        return *this;
    }

    Fixed<T, N>& operator*=(const T & other) {
        for (std::size_t i = 0; i < N; ++i) 
            data[i] *= other;
        return *this;
    }

    T sum() const {
        T ret(0);
        for (std::size_t i = 0; i < N; ++i) 
            ret += data[i];
        return ret;
    }
    
    static Fixed<T, N> random() {
        Fixed<T, N> result;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<T> dis(0.0, 1.0);

        for (std::size_t i = 0; i < N; ++i) {
            result[i] = dis(gen);
        }

        return result;
    }
    
    static const int num_params() {
        return N;
    }

    static const size_t num_bits() {
        return N*sizeof(float)*8;
    }

private:
    T data[N];
};

#endif  // FIXED_H
