#pragma once

#include <vector>
#include <iostream>

struct Matrix
{
	int rows, cols;
	std::vector<double> data;

	Matrix() : rows(0), cols(0) {}

	Matrix(int rows, int cols, double initVal = 0)
		: rows(rows), cols(cols), data(rows * cols, initVal) {}

	double& at(int i, int j)
	{
		if (i < 0 || i >= rows || j < 0 || j >= cols)
			throw std::out_of_range("Index out of range");
		return data[i * cols + j];
	}

	const double& at(int i, int j) const
	{
		if (i < 0 || i >= rows || j < 0 || j >= cols)
			throw std::out_of_range("Index out of range");
		return data[i * cols + j];
	}

	double& operator()(int i, int j) { return at(i, j); }
	const double& operator()(int i, int j) const { return at(i, j); }

	Matrix operator+(const Matrix& other) const;
	Matrix operator-(const Matrix& other) const;
	Matrix operator*(double scalar) const;
	Matrix operator*(const Matrix& other) const;
	Matrix transpose() const;

	void print() const;
};