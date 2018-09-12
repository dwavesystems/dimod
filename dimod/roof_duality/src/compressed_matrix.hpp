/**
# Copyright 2018 D-Wave Systems Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# ================================================================================================
*/
#ifndef COMPRESSED_MATRIX_HPP_INCLUDED
#define COMPRESSED_MATRIX_HPP_INCLUDED

/**
* \file compressed_matrix.hpp
*
* Compressed matrix C++ header file.
*/

#include <vector>
#include <map>
#include <string>

/**
* \namespace sapi
* \brief Namespace for compressed matrix.
*/
namespace compressed_matrix
{
// declaration

/**
* \class CompressedMatrix
* \brief Sparse matrix.
*
* This class stores a (sparse) matrix in compressed-row format.
*
* The matrix is represented by three vectors: \p rowOffsets, \p colIndices, and \p values.
* \p rowOffsets has (fixed) length equal to the number of rows in the matrix, plus one.
* For any row index \p r and \p k such that <tt>rowOffsets[r] <= k < rowOffsets[r+1]</tt>,
* the entry at postition <tt>(r, colIndices[k])</tt> has value <tt>values[k]</tt>.  Column indices are listed in
* increasing order within each row (ie. values in each row are listed from left to right).  Missing column indices
* indicate a value of zero.
*
* Note that nothing prevents you from explicitly storing a zero value but this is wasteful.
*/
// forward declarations
template <typename T>
class CompressedMatrixIterator;

template <typename T>
class CompressedMatrixConstIterator;

template <typename T>
class CompressedMatrix
{
	friend class CompressedMatrixIterator<T>;
	friend class CompressedMatrixConstIterator<T>;
public:
	/**
	* \brief Construct an empty CompressedMatrix of given size.
	*
	* \param numRows number of rows
	* \param numCols number of columns
	*/
	CompressedMatrix(int numRows = 0, int numCols = 0);

	/**
	* \brief Construct a CompressedMatrix from a map.
	*
	* For all valid row indices \p r and column indices \p c, the following is true of the new CompressedMatrix:
	* \code get(r,c) == m[std::make_pair(r,c)] \endcode
	*
	* \param numRows number of rows
	* \param numCols number of columns
	* \param m sparse matrix represented as a map
	* \throw ValueException if any key of \p m has an invalid row or column index
	*/
	CompressedMatrix(int numRows, int numCols,
		const std::map<std::pair<int, int>, T>& m);

	/**
	* \brief Construct a CompressedMatrix from raw compressed-row sparse matrix data.
	*
	* \param numRows number of rows in the matrix
	* \param numCols number of columns in the matrix
	* \param rowOffsets row offset data
	* \param colIndices column index data
	* \param values matrix values
	*
	* Requirements:
	* - <tt>rowOffsets.size() == numRows + 1</tt>
	* - <tt>rowOffsets[0] == 0</tt>
	* - <tt>rowOffsets[i] <= rowOffsets[i + 1]</tt> for all valid \p i
	* - <tt>colIndices[i] < numCols<tt> for all valid \p k
	* - <tt>colIndices[k] < colIndices[k + 1]</tt> whenever <tt>rowIndices[i] <= k < rowIndices[i + 1]</tt>
	*   for all valid \p i
	* - <tt>colIndices.size() == values.size()</tt>
	*/
	CompressedMatrix(int numRows, int numCols,
		std::vector<int> rowOffsets, std::vector<int> colIndices, std::vector<T> values);

	/**
	* \return number of rows
	*/
	int numRows() const;

	/**
	* \return number of columns
	*/
	int numCols() const;

	/**
	* \brief Get the number of nonzero entries in the matrix.
	*
	* Note that if you set an entry to zero, it still counts as a "nonzero entry."  Really, this function counts
	* the number of not-zero-by-default entries.
	*
	* \return number of nonzero entries
	*/
	int nnz() const;

	/**
	* \brief Look up a matrix entry.
	*
	* \param row row index of the entry
	* \param col column index of the entry
	* \return matrix entry (defaults to zero if not found)
	* \throw ValueException if \p row is not less than numRows() or \p col is not less than numCols()
	*/
	T get(int row, int col) const;

	/**
	* \brief Get a reference to matrix entry.
	*
	* Note that this function \em creates an entry (with value zero) if none exists!  Consequently, this function can:
	* - be slow, and
	* - create unwanted zero entries.
	* Use the get() for pure lookups!
	*
	* To minimize shuffling of data, fill the matrix from left to right, then top to bottom.
	*
	* \param row row index of the matrix entry
	* \param col column index of the matrix entry
	* \return reference to the matrix entry
	* \throw ValueException if \p row is not less than numRows() or \p col is not less than numCols()
	*/
	T& operator()(int row, int col);

	/**
	* \brief Get row offset data.
	*
	* This vector always has size <tt>numRows + 1</tt>.
	*
	* \return row offset data
	*/
	const std::vector<int>& rowOffsets() const { return rowOffsets_; }

	/**
	* \brief Get column index data.
	*
	* Values from index rowOffsets()[r] up to (but not including) index rowOffsets()[r + 1] are in strictly
	* increasing order for any valid row index r.
	*
	* \return column index data
	*/
	const std::vector<int>& colIndices() const { return colIndices_; }

	/**
	* \return matrix entry values data
	*/
	const std::vector<T>& values() const { return values_; }

	/**
	* \brief Reserves space in colIndices and values.
	*
	* Calling this function before adding a large number of new entries can save on the number of memory reallocations.
	*
	* \param minValues new minimum size of colIndices and values.
	*/
	void reserve(int minValues);

	/**
	* \p iterator type
	*/
	typedef CompressedMatrixIterator<T> iterator;

	/**
	* \p const_iterator type
	*/
	typedef CompressedMatrixConstIterator<T> const_iterator;

	/**
	* \Returns an iterator referring to the first element in the CompressedMatrix
	*/
	iterator begin();

	/**
	* \Returns a const_iterator referring to the first element in the CompressedMatrix
	*/
	const_iterator begin() const;

	/**
	* \Returns an iterator referring to the past-the-end element in the CompressedMatrix
	*/
	iterator end();

	/**
	* \Returns a const_iterator referring to the past-the-end element in the CompressedMatrix
	*/
	const_iterator end() const;

private:
	int numRows_;
	int numCols_;

	std::vector<int> rowOffsets_;
	std::vector<int> colIndices_;
	std::vector<T> values_;
};


/**
* \brief Converts a CompressedMatrix to a map.
* \param cm the CompressedMatrix to convert
* \return map m where m[make_pair(r,c)] == cm.get(r,c) for all r,c
*/
template <typename T>
std::map<std::pair<int, int>, T> compressedMatrixToMap(const CompressedMatrix<T>& cm);

/**
* \class CompressedMatrixIterator
* \brief Sparse matrix iterator.
*
* This class creates a sparse matrix iterator.
*/
template <typename T>
class CompressedMatrixIterator
{	
public:
	friend class CompressedMatrixConstIterator<T>;
	/**
	* \brief Construct a CompressedMatrix iterator
	*
	* \param cm a CompressedMatrix pointer
	* \param currRow current row
	* \param currIndex current index of colIndices and values
	*/
	CompressedMatrixIterator(CompressedMatrix<T>* cm, int currRow = 0, int currIndex = 0);

	/**
	* \brief Increment the iterator by one
	*/
	void operator++();

	/**
	* \brief Dereference the iterator
	*
	* \return a double reference
	*/
	T& operator*() const;

	/**
	* \brief Get the row of the iterator
	*
	* \return row number
	*/
	int row() const;

	/**
	* \brief Get the column of the iterator
	*
	* \return column number
	*/
	int col() const;

	/**
	* \brief Compare whether two iterators are the same
	*
	* \return true if two iterators are the same; false otherwise
	*/
	bool operator!=(const CompressedMatrixIterator<T>& iter) const;

private:

	CompressedMatrix<T>* cm_;
	int currRow_;
	int currIndex_;	
};

/**
* \class CompressedMatrixConstIterator
* \brief Sparse matrix const_iterator.
*
* This class creates a sparse matrix const_iterator.
*/
template <typename T>
class CompressedMatrixConstIterator
{
public:
	/**
	* \brief Construct a CompressedMatrix const_iterator
	*
	* \param cm a CompressedMatrix pointer
	* \param currRow current row
	* \param currIndex current index of colIndices and values
	*/
	CompressedMatrixConstIterator(const CompressedMatrix<T>* cm, int currRow = 0, int currIndex = 0);

	/**
	* \brief Construct a CompressedMatrixConstIterator from a CompressedMatrixIterator
	*
	* \param iter a CompressedMatrixIterator
	*/
	CompressedMatrixConstIterator(const CompressedMatrixIterator<T>& iter);

	/**
	* \brief Increment the iterator by one
	*/
	void operator++();

	/**
	* \brief Dereference the iterator
	*
	* \return a double value
	*/
	T operator*() const;

	/**
	* \brief Get the row of the iterator
	*
	* \return row number
	*/
	int row() const;

	/**
	* \brief Get the column of the iterator
	*
	* \return column number
	*/
	int col() const;

	/**
	* \brief Compare whether two iterators are the same
	*
	* \return true if two iterators are the same; false otherwise
	*/
	bool operator!=(const CompressedMatrixConstIterator<T>& iter) const;

private:

	const CompressedMatrix<T>* cm_;
	int currRow_;
	int currIndex_;
};


//base exception class
class CompressedMatrixException
{
public:
	CompressedMatrixException(const std::string& m = "compressed matrix exception") : message(m) {}
	const std::string& what() const {return message;}
private:
	std::string message;
};

} //namespace compressed_matrix


// definition

template <typename T>
compressed_matrix::CompressedMatrix<T>::CompressedMatrix(int numRows, int numCols) : numRows_(numRows), numCols_(numCols)
{
	rowOffsets_.resize(numRows_ + 1, 0);
}

template <typename T>
compressed_matrix::CompressedMatrix<T>::CompressedMatrix(int numRows, int numCols, const std::map<std::pair<int, int>, T>& m) : numRows_(numRows), numCols_(numCols)
{
	int offset = 0;
	int currRow = 0;
	rowOffsets_.resize(numRows_ + 1);
	for (typename std::map<std::pair<int, int>, T>::const_iterator it = m.begin(), end = m.end(); it != end; ++it)
	{
		if (it->first.first >= numRows_ || it->first.second >= numCols_)
			throw CompressedMatrixException("map contains invalid index for compressed matrix constructor.");

		if (currRow <= it->first.first)
		{
			for (int i = currRow; i <= it->first.first; i++)
				rowOffsets_[i] = offset;
			currRow = it->first.first + 1;
		}

		colIndices_.push_back(it->first.second);
		values_.push_back(it->second);
		++offset;
	}
	for (int i = currRow; i < rowOffsets_.size(); i++)
		rowOffsets_[i] = offset;
}

template <typename T>
compressed_matrix::CompressedMatrix<T>::CompressedMatrix(int numRows, int numCols, std::vector<int> rowOffsets, std::vector<int> colIndices, std::vector<T> values) : numRows_(numRows), numCols_(numCols)
{
	if (rowOffsets.size() != numRows_ + 1)
		throw CompressedMatrixException("row offset vector's size is incorrect.");
	if (colIndices.size() != values.size())
		throw CompressedMatrixException("column indices vector's size and values vector's size are different.");

	rowOffsets_ = rowOffsets;
	colIndices_ = colIndices;
	values_ = values;
}

template <typename T>
int compressed_matrix::CompressedMatrix<T>::numRows() const
{
	return numRows_;
}

template <typename T>
int compressed_matrix::CompressedMatrix<T>::numCols() const
{
	return numCols_;
}

template <typename T>
int compressed_matrix::CompressedMatrix<T>::nnz() const
{
	return static_cast<int>(values_.size());
}

template <typename T>
T compressed_matrix::CompressedMatrix<T>::get(int row, int col) const
{
	if (row >= numRows_ || col >= numCols_)
		throw CompressedMatrixException("row or col contains invalid index for compressed matrix get.");

	int start = rowOffsets_[row];
	int end = rowOffsets_[row + 1];

	if (start == end)
		return 0;

	int low = start;
	int high = end - 1;
	int mid;
	while (low <= high)
	{
		mid = low + (high - low) / 2;
		if (colIndices_[mid] == col)
			return values_[mid];
		else if (colIndices_[mid] < col)
			low = mid + 1;
		else
			high = mid - 1;
	}

	return 0;
}


template <typename T>
T& compressed_matrix::CompressedMatrix<T>::operator()(int row, int col)
{
	if (row >= numRows_ || col >= numCols_)
		throw CompressedMatrixException("row or col contains invalid index for compressed matrix operator().");

	int start = rowOffsets_[row];
	int end = rowOffsets_[row + 1];	

	if (start == end) //not found
	{
		for (int i = row + 1; i < rowOffsets_.size(); i++)
			++rowOffsets_[i];
		colIndices_.insert(colIndices_.begin() + start, col);
		values_.insert(values_.begin() + start, 0);
		return values_[start];
	}

	int low = start;
	int high = end - 1;
	int mid;
	while (low <= high)
	{
		mid = low + (high - low) / 2;
		if (colIndices_[mid] == col)
			return values_[mid]; //found, return the reference
		else if (colIndices_[mid] < col)
			low = mid + 1;
		else
			high = mid - 1;
	}

	//not found
	for (int i = row + 1; i < rowOffsets_.size(); i++)
		++rowOffsets_[i];
	if (colIndices_[mid] < col)
		++mid;
	colIndices_.insert(colIndices_.begin() + mid, col);
	values_.insert(values_.begin() + mid, 0);
	return values_[mid];
}

template <typename T>
void compressed_matrix::CompressedMatrix<T>::reserve(int minValues)
{
	colIndices_.reserve(minValues);
	values_.reserve(minValues);
}

template <typename T>
typename compressed_matrix::CompressedMatrix<T>::const_iterator compressed_matrix::CompressedMatrix<T>::begin() const
{
	//find the last 0 in rowOffsets_[]; there will always be at least one 0 in rowOffsets_[], so doesn't need to check if rowOffsets_[low] is 0 or not
	int low = 0;
	int high = numRows_;
	int mid;
	while (low < high)
	{		
		mid = low + (high - low + 1) / 2;
		if (rowOffsets_[mid] == 0)
			low = mid;
		else
			high = mid - 1;
	}

	return CompressedMatrixConstIterator<T>(this, low, 0);
}

template <typename T>
typename compressed_matrix::CompressedMatrix<T>::const_iterator compressed_matrix::CompressedMatrix<T>::end() const
{
	return CompressedMatrixConstIterator<T>(this, numRows_);
}

template <typename T>
typename compressed_matrix::CompressedMatrix<T>::iterator compressed_matrix::CompressedMatrix<T>::begin()
{
	//find the last 0 in rowOffsets_[]; there will always be at least one 0 in rowOffsets_[], so doesn't need to check if rowOffsets_[low] is 0 or not
	int low = 0;
	int high = numRows_;
	int mid;
	while (low < high)
	{		
		mid = low + (high - low + 1) / 2;
		if (rowOffsets_[mid] == 0)
			low = mid;
		else
			high = mid - 1;
	}

	return CompressedMatrixIterator<T>(this, low, 0);	  
}

template <typename T>
typename compressed_matrix::CompressedMatrix<T>::iterator compressed_matrix::CompressedMatrix<T>::end()
{
	return CompressedMatrixIterator<T>(this, numRows_);
}


template <typename T>
std::map<std::pair<int, int>, T> compressed_matrix::compressedMatrixToMap(const CompressedMatrix<T>& cm)
{
	std::map<std::pair<int, int>, T> ret;
	for (int i = 0; i < cm.numRows(); i++)
	{
		int start = cm.rowOffsets()[i];
		int end = cm.rowOffsets()[i + 1];
		for (int j = start; j < end; j++)
			ret.insert(std::make_pair(std::make_pair(i, cm.colIndices()[j]), cm.values()[j]));
	}

	return ret;
}


template <typename T>
compressed_matrix::CompressedMatrixIterator<T>::CompressedMatrixIterator(CompressedMatrix<T>* cm, int currRow, int currIndex) : cm_(cm), currRow_(currRow), currIndex_(currIndex)
{

}

template <typename T>
void compressed_matrix::CompressedMatrixIterator<T>::operator++()
{		
	++currIndex_;
	if (currIndex_ == cm_->rowOffsets_[currRow_ + 1])
	{
		//++currRow_;
		//while (currRow_<cm_->numRows_ && cm_->rowOffsets_[currRow_]==cm_->rowOffsets_[currRow_+1])
		//	++currRow_;
		//find the last element which is equal to cm_->rowOffsets_[currRow_+1]
		int target = cm_->rowOffsets_[currRow_ + 1];
		int low = currRow_ + 1;
		int high = cm_->numRows_;
		int mid;
		while (low < high)
		{
			mid = low + (high - low + 1) / 2;
			if (cm_->rowOffsets_[mid] == target)
				low = mid;
			else
				high = mid - 1;
		}

		currRow_ = low;
	}
}

template <typename T>
T& compressed_matrix::CompressedMatrixIterator<T>::operator*() const
{
	return cm_->values_[currIndex_];
}

template <typename T>
int compressed_matrix::CompressedMatrixIterator<T>::row() const
{
	return currRow_;
}

template <typename T>
int compressed_matrix::CompressedMatrixIterator<T>::col() const
{
	return cm_->colIndices_[currIndex_];
}

template <typename T>
bool compressed_matrix::CompressedMatrixIterator<T>::operator!=(const CompressedMatrixIterator<T>& iter) const
{
	if (cm_ != iter.cm_)
		throw CompressedMatrixException("compare iterators from two different CompressedMatrix objects.");

	return (currRow_ != iter.currRow_);
}

template <typename T>
compressed_matrix::CompressedMatrixConstIterator<T>::CompressedMatrixConstIterator(const CompressedMatrix<T>* cm, int currRow, int currIndex) : cm_(cm), currRow_(currRow), currIndex_(currIndex)
{

}

template <typename T>
compressed_matrix::CompressedMatrixConstIterator<T>::CompressedMatrixConstIterator(const compressed_matrix::CompressedMatrixIterator<T>& iter) : cm_(iter.cm_), currRow_(iter.currRow_), currIndex_(iter.currIndex_)
{

}

template <typename T>
void compressed_matrix::CompressedMatrixConstIterator<T>::operator++()
{		
	++currIndex_;
	if (currIndex_ == cm_->rowOffsets_[currRow_ + 1])
	{
		//++currRow_;
		//while (currRow_<cm_->numRows_ && cm_->rowOffsets_[currRow_]==cm_->rowOffsets_[currRow_+1])
		//	++currRow_;
		//find the last element which is equal to cm_->rowOffsets_[currRow_+1]
		int target = cm_->rowOffsets_[currRow_ + 1];
		int low = currRow_ + 1;
		int high = cm_->numRows_;
		int mid;
		while (low < high)
		{
			mid = low + (high - low + 1) / 2;
			if (cm_->rowOffsets_[mid] == target)
				low = mid;
			else
				high = mid - 1;
		}

		currRow_ = low;
	}
}

template <typename T>
T compressed_matrix::CompressedMatrixConstIterator<T>::operator*() const
{
	return cm_->values_[currIndex_];
}

template <typename T>
int compressed_matrix::CompressedMatrixConstIterator<T>::row() const
{
	return currRow_;
}

template <typename T>
int compressed_matrix::CompressedMatrixConstIterator<T>::col() const
{
	return cm_->colIndices_[currIndex_];
}

template <typename T>
bool compressed_matrix::CompressedMatrixConstIterator<T>::operator!=(const CompressedMatrixConstIterator<T>& iter) const
{
	if (cm_ != iter.cm_)
		throw CompressedMatrixException("compare iterators from two different CompressedMatrix objects.");

	return (currRow_ != iter.currRow_);
}

#endif // COMPRESSED_MATRIX_HPP_INCLUDED
