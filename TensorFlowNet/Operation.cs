using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public abstract class Operation
    {
        public string Identifier { get; internal set; }
        
        internal Operation()
        {

        }

        public abstract string OperationName { get; }

        public abstract void Execute();

        public override string ToString()
        {
            return $"<Operation \"{Identifier}\" type=NoOp>";
        }
    }

    public class GlobalVariablesInitializerOperation : Operation
    {
        IEnumerable<VariableTensor> mGlobalVariables { get; set; }

        public override string OperationName => "Init";

        public GlobalVariablesInitializerOperation(IEnumerable<VariableTensor> globalVariables)
        {
            mGlobalVariables = globalVariables;
        }

        public override void Execute()
        {
            foreach (VariableTensor tensor in mGlobalVariables)
            {
                tensor.Initialize();
            }
        }
    }

    public class GradientDescentOperation : Operation
    {
        public GradientDescentOperation(Tensor loss)
        {

        }

        public override string OperationName => "GradientDescent";

        public override void Execute()
        {
            //new ComputeGradientsOperation().Execute();
            //new ApplyGradientsOperation().Execute();
        }
    }

    public class ComputeGradientsOperation : Operation
    {
        public override string OperationName => "ComputeGradients";

        public override void Execute()
        {
            throw new NotImplementedException();
        }
    }

    public class ApplyGradientsOperation : Operation
    {
        List<VariableTensor> mVariables { get; set; }
        float mGradientQuantity { get; set; }

        public ApplyGradientsOperation(List<VariableTensor> variables, float gradientQuantity)
        {
            mGradientQuantity = gradientQuantity;
            mVariables = variables;
        }

        public override string OperationName => "ApplyGradients";

        public override void Execute()
        {
            mVariables.ForEach(x => x.Value += mGradientQuantity);
        }
    }
}
