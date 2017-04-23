using System;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using MathNet.Numerics.LinearAlgebra;

namespace TensorFlowNet.VSTests
{
    /// <summary>
    /// Tests derived from the "Get Started" documentation at https://www.tensorflow.org/get_started/get_started
    /// </summary>
    [TestClass]
    public class GetStartedTests
    {
        [TestMethod]
        public void TestConstantNode()
        {
            var node = TensorFlow.Constant<float>(3.0f);
            var expected = new Matrix<float>[] { Matrix<float>.Build.Dense(1, 1, 3.0f) };

            //TensorFlow.Session().Run(node)[0][0, 0] == expected[0][0, 0]
            //true
            //Assert.AreEqual(expected, TensorFlow.Session().Run(node));
            Assert.IsTrue(expected.MatrixArraysAreEqual(TensorFlow.Session().Run(node)));
        }

        [TestMethod]
        public void TestConstantNodesWithNoOp()
        {
            var node1 = TensorFlow.Constant<float>(3.0f);
            var node2 = TensorFlow.Constant(4.0f);
            var sess = TensorFlow.Session();
            Assert.IsTrue(new Matrix<float>[] { Matrix<float>.Build.Dense(1, 1, 3.0f),
                                                Matrix<float>.Build.Dense(1, 1, 4.0f)}.MatrixArraysAreEqual(sess.Run(node1, node2)));
        }

        [TestMethod]
        public void TestAdditionOperation()
        {
            var node1 = TensorFlow.Constant<float>(3.0f);
            var node2 = TensorFlow.Constant(4.0f);
            var sess = TensorFlow.Session();
            var node3 = TensorFlow.Add(node1, node2);
            Assert.IsTrue(new Matrix<float>[] { Matrix<float>.Build.Dense(1, 1, 7.0f) }.MatrixArraysAreEqual(TensorFlow.Session().Run(node3)));
        }
    }
}
