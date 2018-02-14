using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TensorFlowNet.Demo
{
    class Program
    {
        static void Main(string[] args)
        {
            // Create a constant float tensor with value 3.0
            var node1 = TensorFlow.Constant<float>(3.0f);
            // Create a constant float tensor with value 4.0
            var node2 = TensorFlow.Constant(4.0f);

            Print(node1);
            Print(node2);

            // Create a new Session
            var sess = TensorFlow.Session();
            // Run the session and print the result
            // [3], [4]
            Print(sess.Run(node1, node2));

            // Create a new addition tensor using the Add method
            var node3 = TensorFlow.Add(node1, node2);
            // Show the details of the resulting Add tensor
            Print(node3);
            // Run the session on the add tensor to evaluate it, and print the result
            // [7]
            Print(sess.Run(node3));
            // Test implicit operator creation of Addition tensor, and print the result
            // [7]
            Print(sess.Run(node1 + node2));

            // Create two new placeholder tensors
            var a = TensorFlow.Placeholder(typeof(float));
            var c = TensorFlow.Placeholder(typeof(float));
            // Create an addition tensor using the two placeholders
            var adder_node = a + c;
            // Show the details of the resulting addition tensor
            Print(adder_node);
            // Run the session on the addition tensor to evaluate it, passing in single float values for the placeholders
            // which are found by their identifier
            // [7.5]
            Print(sess.Run(
                adder_node, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.Dense(1, 1, 3.0f),
                    [c.Identifier] = Matrix<float>.Build.Dense(1, 1, 4.5f)
                }));
            // Run another session on the addition tensor to evaluate it, passing in matrix values for the placeholders
            // [3 7]
            Print(sess.Run(
                adder_node, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 3 } }),
                    [c.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 2, 4 } })
                }));

            // Take the addition tensor from earlier and multiply it by three to create a multiplication tensor
            var add_and_triple = adder_node * 3;
            // Run the multiplication tensor (using the same placeholder tensors from earlier), and print the result
            // [22.5]
            Print(sess.Run(
                add_and_triple, new Dictionary<string, Matrix<float>>
                {
                    [a.Identifier] = Matrix<float>.Build.Dense(1, 1, 3.0f),
                    [c.Identifier] = Matrix<float>.Build.Dense(1, 1, 4.5f)
                }));

            // Create a Squaring tensor, run the session and print the value
            // [9]
            Print(sess.Run(TensorFlow.Square((ConstantTensor)3.0)));

            // Create tensor containing a 2 row by 3 column matrix of just 1s
            // |1 1 1|
            // |1 1 1|
            var t = TensorFlow.Constant(Matrix<float>.Build.Dense(2, 3, 1));
            // Create a reduce sum tensor with no axis which sums the values of the entire matrix
            // [6]
            Print(sess.Run(TensorFlow.ReduceSum(t)));
            // Create a reduce sum tensor with the Y axis which reduces the number of rows by adding
            // the values down a column
            // [2 2 2]
            Print(sess.Run(TensorFlow.ReduceSum(t, ReduceSumAxis.Y)));
            // Create a reduce sum tensor with the X axis which reduces the number of columns by adding
            // the values across a row
            // [3 3]
            Print(sess.Run(TensorFlow.ReduceSum(t, ReduceSumAxis.X)));

            
            // Regression Models

            // Create variable tensors with initial values
            var W = TensorFlow.Variable(0.3f);
            var b = TensorFlow.Variable(-0.3f);
            // Create a placeholder tensor that can be defined when the session starts
            var x = TensorFlow.Placeholder(typeof(float));
            // Create a graph of tensors that represents an equation
            var linear_model = W * x + b;

            // Required to initialize all variable tensors to their initial values
            var init = TensorFlow.GlobalVariablesInitializer();
            sess.Run(init);

            // Run the linear model against an initial vector
            // [0 0.3 0.6 0.9]
            Print(sess.Run(linear_model, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4} })
            }));

            // To optimize a model, define loss using mean square error
            var y = TensorFlow.Placeholder(typeof(float));
            var squared_deltas = TensorFlow.Square(linear_model - y);
            var loss = TensorFlow.ReduceSum(squared_deltas);
            // Run the loss tensor to produce a loss value
            // [23.66]
            Print(sess.Run(loss, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4} }),
                [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3} })
            }));

            // Reassign values for W and b, then run again to produce loss value
            var fixW = TensorFlow.Assign(W, -1.0f);
            var fixb = TensorFlow.Assign(b, 1.0f);
            // [-1], [1]
            Print(sess.Run(W, b));
            // [0]
            Print(sess.Run(loss, new Dictionary<string, Matrix<float>>
            {
                [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4 } }),
                [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3 } })
            }));

            // Training a Model

            // Define optimizers which incrementally change variables in order to minimize loss
            // Gradient descent optimizer modifies each variable according to the magnitude of the derivative of loss
            // with respect to that variable
            var optimizer = TensorFlow.Train.GradientDescentOptimizer(0.01f);
            // Train is an operation not a tensor
            var train = optimizer.Minimize(loss);

            // reset variable values to (incorrect) defaults to run again
            sess.Run(init);
            // Training 
            // Since 'train' is an operation, not a tensor, it doesn't return a value when run,
            // it just modifies it's own variables (in this case, W and b)
            for (int i = 0; i < 1000; i++)
            {
                sess.Run(train, new Dictionary<string, Matrix<float>>
                {
                    [x.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 1, 2, 3, 4 } }),
                    [y.Identifier] = Matrix<float>.Build.DenseOfArray(new float[,] { { 0, -1, -2, -3 } })
                });
            }
            // Now that the model has been trained, evaluate W and b to see what values they wer
            // assigned by the training
            // 
            Print(sess.Run(W, b));


            // Additional Tensor Tests

            // Create a constant tensor and cube it to test powers
            var q = TensorFlow.Constant(2.0f);
            // 2 ^ 3
            var q_cubed = TensorFlow.Pow(q, (ConstantTensor)3);
            // [8]
            Print(sess.Run(q_cubed));

            // Test taking the derivative of the tensor to get a new tensor, then evaluating it
            // 3 * (2 ^ 2)
            var q_cubed_derived = q_cubed.Derive();
            // [12]
            Print(sess.Run(q_cubed_derived));

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
