#include "Matrix.hpp"
#include <omp.h>

Matrix Matrix::operator+(const Matrix& other) const
{
    if ((rows == other.rows && cols == other.cols) ||
        (rows == other.rows && other.cols == 1) ||
        (cols == other.cols && other.rows == 1)) {
        
        Matrix result(rows, cols);
        int total = rows * cols;
        #pragma omp parallel for
        for (int idx = 0; idx < total; idx++)
        {
            int i = idx / cols;
            int j = idx % cols;
            double other_val;
            if (other.rows == rows && other.cols == cols)
                other_val = other.data[idx];
            else if (other.cols == 1)
                other_val = other.data[i];
            else
                other_val = other.data[j];
            result.data[idx] = data[idx] + other_val;
        }
        return result;
    }
    else
        throw std::invalid_argument("Matrix dimensions do not match for addition");
}

Matrix Matrix::operator-(const Matrix& other) const
{
    if ((rows == other.rows && cols == other.cols) ||
        (rows == other.rows && other.cols == 1) ||
        (cols == other.cols && other.rows == 1)) {
        
        Matrix result(rows, cols);
        int total = rows * cols;
        #pragma omp parallel for
        for (int idx = 0; idx < total; idx++)
        {
            int i = idx / cols;
            int j = idx % cols;
            double other_val;
            if (other.rows == rows && other.cols == cols)
                other_val = other.data[idx];
            else if (other.cols == 1)
                other_val = other.data[i];
            else
                other_val = other.data[j];
            result.data[idx] = data[idx] - other_val;
        }
        return result;
    }
    else
        throw std::invalid_argument("Matrix dimensions do not match for subtraction");
}

Matrix Matrix::operator*(double scalar) const
{
    Matrix result(rows, cols);
    int total = rows * cols;
    #pragma omp parallel for
    for (int idx = 0; idx < total; idx++)
        result.data[idx] = data[idx] * scalar;
    
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const
{
    if (cols != other.rows)
        throw std::invalid_argument("Matrix dimensions do not allow multiplication");
    
    Matrix result(rows, other.cols);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < other.cols; j++)
        {
            double sum = 0;
            for (int k = 0; k < cols; k++)
                sum += (*this)(i, k) * other(k, j);
            
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix Matrix::transpose() const
{
    Matrix result(cols, rows);
    #pragma omp parallel for
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            result(j, i) = (*this)(i, j);

    return result;
}

void Matrix::print() const
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            std::cout << (*this)(i, j) << " ";
        }
        std::cout << "\n";
    }
}