using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            var node1 = TensorFlow.Constant<float>(3.0f);
            var node2 = TensorFlow.Constant(4.0f);

            Print(node1);
            Print(node2);

            var sess = TensorFlow.Session();
            // [3], [4]
            Print(sess.Run(node1, node2));


            var node3 = TensorFlow.Add(node1, node2);
            Print(node3);
            // [7]
            Print(sess.Run(node3));

            var a = TensorFlow.Placeholder(typeof(float));
            var c = TensorFlow.Placeholder(typeof(float));
            var adder_node = a + c;
            Print(adder_node);
            // [7.5]
            Print(sess.Run(
                adder_node, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.Dense(1, 1, 3.0f),
                    [c.Identifier] = Matrix<float>.Build.Dense(1, 1, 4.5f)
                }));
            // [3 7]
            Print(sess.Run(
                adder_node, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 3 } }),
                    [c.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 2, 4 } })
                }));

            var add_and_triple = adder_node * 3;
            // [22.5]
            Print(sess.Run(
                add_and_triple, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.Dense(1, 1, 3.0f),
                    [c.Identifier] = Matrix<float>.Build.Dense(1, 1, 4.5f)
                }));

            // [9]
            Print(sess.Run(TensorFlow.Square((ConstantTensor)3.0)));

            var t = TensorFlow.Constant(Matrix<float>.Build.Dense(2, 3, 1));
            // [6]
            Print(sess.Run(TensorFlow.ReduceSum(t)));
            // [2 2 2]
            Print(sess.Run(TensorFlow.ReduceSum(t, ReduceSumAxis.Y)));
            // [3 3]
            Print(sess.Run(TensorFlow.ReduceSum(t, ReduceSumAxis.X)));

            var W = TensorFlow.Variable(0.3f);
            var b = TensorFlow.Variable(-0.3f);
            var x = TensorFlow.Placeholder(typeof(float));
            var linear_model = W * x + b;

            var init = TensorFlow.GlobalVariablesInitializer();
            sess.Run(init);

            // [0 0.3 0.6 0.9]
            Print(sess.Run(linear_model, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4} })
            }));

            var y = TensorFlow.Placeholder(typeof(float));
            var squared_deltas = TensorFlow.Square(linear_model - y);
            var loss = TensorFlow.ReduceSum(squared_deltas);
            
            // [23.66]
            Print(sess.Run(loss, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4} }),
                [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3} })
            }));

            var fixW = TensorFlow.Assign(W, -1.0f);
            var fixb = TensorFlow.Assign(b, 1.0f);
            
            // [0]
            Print(sess.Run(loss, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4 } }),
                [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3 } })
            }));

            var optimizer = TensorFlow.Train.GradientDescentOptimizer(0.01f);
            var train = optimizer.Minimize(loss);

            // reset variable values to (incorrect) defaults
            sess.Run(init);
            for (int i = 0; i < 1000; i++)
            {
                sess.Run(train, new Dictionary<string, Matrix<float>>
                {
                    [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4 } }),
                    [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3 } })
                });
            }
            Print(sess.Run(W, b));

            Console.WriteLine("Done");
            Console.ReadLine();
        }
        public static string FormatArrayForPrint(object[] array)
        {
            return $"[{String.Join(", ", array)}]";
        }

        public static void Print(object[] array)
        {
            Console.WriteLine(FormatArrayForPrint(array));
        }

        public static void Print(Tensor tensor)
        {
            Console.WriteLine(tensor.ToString());
        }

        public static void Print(Matrix<float>[] matrices)
        {
            Console.WriteLine(String.Join(", ", matrices.Select(x => MatrixToString(x))));
        }

        public static void Print(Matrix<float> matrix)
        {
            Console.WriteLine(MatrixToString(matrix));
        }

        public static string MatrixToString(Matrix<float> matrix)
        {
            StringBuilder sb = new StringBuilder();
            for(int i = 0; i < matrix.RowCount; i++)
            {
                if (i == 0)
                {
                    sb.Append("[");
                }
                else
                {
                    sb.Append("| ");
                }

                List<string> rowLines = new List<string>();
                for (int j = 0; j < matrix.ColumnCount; j++)
                {
                    rowLines.Add(matrix[i, j].ToString());
                }

                if (i == matrix.RowCount - 1)
                {
                    sb.Append(String.Join(" ", rowLines) + "]");
                }
                else
                {
                    sb.AppendLine(String.Join(" ", rowLines) + " |");
                }
            }
            return sb.ToString();
        }

        public static string VectorToString(double[] vector)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append("[");

            foreach (double val in vector)
            {
                sb.AppendLine(String.Concat("[", val, "]"));
            }

            sb.Append("]");
            return sb.ToString();
        }
    }
}
