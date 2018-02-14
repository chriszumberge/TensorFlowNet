using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TensorFlowNet
{
    public abstract class Operation : Node
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
        public override string OperationName => "Init";
        public override string GraphText => OperationName;

        Func<IEnumerable<VariableTensor>> _globalVariablesGetterFunction { get; set; }

        public GlobalVariablesInitializerOperation(Func<IEnumerable<VariableTensor>> globalVariablesGetterFunction)
        {
            _globalVariablesGetterFunction = globalVariablesGetterFunction;
        }

        public override void Execute()
        {
            foreach (VariableTensor tensor in _globalVariablesGetterFunction())
            {
                tensor.Initialize();
            }
        }
    }

    public class GradientDescentOperation : Operation
    {
        public override string OperationName => "GradientDescent";
        public override string GraphText => "Op";

        Tensor _lossTensor;
        List<VariableTensor> _varList;

        /// <summary>
        /// Initializes a new instance of the <see cref="GradientDescentOperation" /> class.
        /// </summary>
        /// <param name="loss">A Tensor containing the value to minimize.</param>
        /// <param name="varList">Optional list VariableTensors to update to minimize loss.</param>
        public GradientDescentOperation(Tensor loss, List<VariableTensor> varList)
        {
            _lossTensor = loss;
            _varList = varList;
        }

        // Just calls compute_gradients, then apply_gradients.. 
        // theoretically should this actually create them as nodes?
        public override void Execute()
        {
            var computeOp = new ComputeGradientsOperation(_lossTensor, _varList);
            computeOp.Execute();

            new ApplyGradientsOperation(computeOp.ComputedGradients).Execute();
        }
    }

    public class ComputeGradientsOperation : Operation
    {
        public override string OperationName => "ComputeGradients";
        public override string GraphText => "Op";

        Tensor _lossTensor;
        List<VariableTensor> _varList;

        public ComputeGradientsOperation(Tensor loss, List<VariableTensor> varList)
        {
            _lossTensor = loss;
            _varList = varList;
        }

        List<GradientVariablePair> _computedGradients { get; set; } = new List<GradientVariablePair>();
        public List<GradientVariablePair> ComputedGradients => _computedGradients;

        public override void Execute()
        {
            
        }
    }

    public class ApplyGradientsOperation : Operation
    {
        public override string OperationName => "ApplyGradients";
        public override string GraphText => "Op";

        List<GradientVariablePair> _gradients;

        public ApplyGradientsOperation(List<GradientVariablePair> gradients)
        {
            _gradients = gradients;
        }

        public override void Execute()
        {
            _gradients.ForEach(x => x.Variable.Value += x.Gradient);
        }
    }

    public class GradientVariablePair : Tuple<float, VariableTensor>
    {
        public GradientVariablePair(float gradient, VariableTensor variable) : base(gradient, variable) { }

        public float Gradient => Item1;
        public VariableTensor Variable => Item2;
    }
}
