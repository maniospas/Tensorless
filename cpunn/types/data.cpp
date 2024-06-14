#ifndef DATA_H
#define DATA_H

#include <iostream>
#include <vector>
#include <bitset>
#include <cstdlib>
#include <random>
#include "vecutils.cpp"


class DATA {
private:
    VECTOR value;
    VECTOR value1;
    VECTOR value2;
public:
    explicit DATA(int v) : value(v) {}
    explicit DATA(long v) : value(v) {}
    explicit DATA(long long v) : value(v) {}
    explicit DATA(int v, int v1, int v2) : value(v), value1(v1), value2(v2) {}
    explicit DATA(long v, long v1, long v2) : value(v), value1(v1), value2(v2) {}
    explicit DATA(long long v, long long v1, long long v2) : value(v), value1(v1), value2(v2) {}
    DATA(const DATA &other) : value(other.value), value1(other.value1), value2(other.value2) {}
    DATA() : value(0) {}

    DATA(double) = delete;
    DATA(float) = delete;
    DATA(unsigned int) = delete;

    DATA& operator=(int) = delete;
    DATA& operator=(double) = delete;
    DATA& operator=(float) = delete;
    DATA& operator=(long) = delete;
    DATA& operator=(unsigned int) = delete;
    DATA& operator=(unsigned long) = delete;
    DATA& operator=(unsigned long long) = delete;

    VECTOR lower() {
        return value;
    }

    DATA& operator=(const DATA &other) {
        if (this != &other) {
            value = other.value;
            value1 = other.value1;
            value2 = other.value2;
        }
        return *this;
    }

    void assert_simple() const {
        if(value1 || value2) 
            throw std::logic_error("cannot typecast complicated data");
    }

    explicit operator VECTOR() const {
        assert_simple();
        return value;
    }
    
    explicit operator bool() const {
        return (bool)(value) || (bool)value1;
    }

    friend std::ostream& operator<<(std::ostream &os, const DATA &si) {
        std::cout << std::bitset<(sizeof(VECTOR)*8)>(si.value1) << "_" << std::bitset<(sizeof(VECTOR)*8)>(si.value);
        return os;
    }

    const DATA& print(const std::string& text="") const {
        std::cout << text << *this << "\n";
        return *this;
    }

    const bool isZeroAt(int i) {
        VECTOR a = value | value1 | value2;
        return (a >> i) & 1;
    }

    const int get(int i) {
        return ((value >> i) & 1) + ((value1 >> i) & 1)*2 + ((value2 >> i) & 1)*4;
    }

    const int get(int i) const {
        return ((value >> i) & 1) + ((value1 >> i) & 1)*2 + ((value2 >> i) & 1)*4;
    }

    const DATA& set(int i, int val) {
        if(sizeof(VECTOR)*8<=i || i<0)
            throw std::logic_error("out of of range");
        if(val<0 || val>7)
            throw std::logic_error("can only set values {0,1,2,3}");
        if(val&1)
            value |= ONEHOT(i);
        else
            value &= ~ONEHOT(i);
        if(val&2)
            value1 |= ONEHOT(i);
        else
            value1 &= ~ONEHOT(i);
        if(val&4)
            value2 |= ONEHOT(i);
        else
            value2 &= ~ONEHOT(i);
        return *this;
    }

    int operator[](int i) {
        return get(i);
    }

    int operator[](int i) const {
        return get(i);
    }
    
    DATA& operator[](std::pair<int, int> p) {
        set(p.first, p.second);
        return *this;
    }

    int countNonZeros() { // WARNING: this is not parallelized
        VECTOR n = value | value1 | value2;
        VECTOR count = 0;
        while (n) {
            count += n & 1; // Add the least significant bit
            n >>= 1;        // Shift the number right by 1 bit
        }
        return count;
    }
    
    DATA operator~() const {
        return DATA(~(value | value1 | value2));
    }
    DATA operator!=(const DATA &other) const {
        return DATA((other.value^value) | (other.value1^value1) | (other.value2^value2));
    }
    DATA operator*(const DATA &other) const {
        return DATA(other.value&value, 
                    (other.value1&value) | (other.value&value1),
                    (other.value2&value) | (other.value&value2) | (other.value1&value1));
    }
    DATA operator+(const DATA &other) const {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        return DATA(other.value^value, 
                    other.value1^value1^carry,
                    other.value2^value2^carry1
                    );
    }
    DATA operator>>(const int other) const {
        assert_simple();
        return DATA(value>>other);
    }
    DATA operator<<(const int other) const {
        assert_simple();
        return DATA(value<<other);
    }
    const DATA& operator+=(const DATA &other) {
        VECTOR carry = other.value&value;
        VECTOR carry1 = (value1 & other.value1) | (carry & (value1 ^ other.value1));
        value = other.value^value;
        value1 = other.value1^value1^carry;
        value2 = other.value2^value2^carry1;
        return *this;
    }
    const DATA& operator*=(const DATA &other) {
        value2 = (other.value2&value) | (other.value&value2) | (other.value1&value1);
        value1 = (other.value1&value) | (other.value&value1);
        value = other.value&value; 
        return *this;
    }
};


DATA vec2data(const std::vector<int>& binaryVector) {
    VECTOR result = 0;
    for (int i = 0; i < binaryVector.size(); ++i) 
        if (binaryVector[i]) 
            result |= (1 << i);
    return DATA(result);
}

VECTOR inline randvec(int llr) {
    VECTOR ret = lrand();
    for(int i=0;i<llr;++i)
        ret &= lrand();
    return ret;
}

DATA inline rand(int llr) {
    return DATA(randvec(llr), randvec(llr), randvec(llr));
}

#endif // DATA_H
