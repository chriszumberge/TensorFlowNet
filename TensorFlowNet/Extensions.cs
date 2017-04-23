using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public static class Extensions
    {
        public static bool MatricesAreEqual<T>(this Matrix<T> source, Matrix<T> compare) where T : struct, IEquatable<T>, IFormattable
        {
            if (source.RowCount != compare.RowCount || source.ColumnCount != compare.ColumnCount)
            {
                return false;
            }

            for (int rowCount = 0; rowCount < source.RowCount; rowCount++)
            {
                for (int colCount = 0; colCount < source.ColumnCount; colCount++)
                {
                    if (!source[rowCount, colCount].Equals(compare[rowCount, colCount]))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        public static bool MatrixArraysAreEqual<T>(this Matrix<T>[] sourceArray, Matrix<T>[] compareArray) where T : struct, IEquatable<T>, IFormattable
        {
            if (sourceArray.Length != compareArray.Length)
            {
                return false;
            }

            for (int matrixCount = 0; matrixCount < sourceArray.Length; matrixCount++)
            {
                var source = sourceArray[matrixCount];
                var compare = compareArray[matrixCount];

                if (!source.MatricesAreEqual(compare))
                {
                    return false;
                }
            }

            return true;
        }
    }
}
