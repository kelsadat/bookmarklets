class Matrix {
	static dot(matrixA, matrixB) {
	  const numRowsA = matrixA.length;
	  const numColsA = matrixA[0].length;
	  const numRowsB = matrixB.length;
	  const numColsB = matrixB[0].length;

	  if (numColsA !== numRowsB) {
		throw new Error('Number of columns in matrixA must be equal to the number of rows in matrixB.');
	  }

	  const result = new Array(numRowsA);

	  for (let i = 0; i < numRowsA; i++) {
		result[i] = new Array(numColsB);
		for (let j = 0; j < numColsB; j++) {
		  result[i][j] = 0;
		  for (let k = 0; k < numColsA; k++) {
			result[i][j] += matrixA[i][k] * matrixB[k][j];
		  }
		}
	  }

	  return result;
	}

	static multiplyInt(matrix, integer) {
	  const numRows = matrix.length;
	  const numCols = matrix[0].length;
	  // Create a new matrix to store the result
	  const resultMatrix = new Array(numRows);

	  for (let i = 0; i < numRows; i++) {
		resultMatrix[i] = new Array(numCols);
		for (let j = 0; j < numCols; j++) {
		  resultMatrix[i][j] = matrix[i][j] * integer;
		}
	  }

	  return resultMatrix;
	}

	static multiply(array1, array2) {
		if (array1.length !== array2.length || array1[0].length !== array2[0].length) {
			throw new Error('Arrays must have the same dimensions for element-wise multiplication.');
		}

		const numRows = array1.length;
		const numCols = array1[0].length;
		const resultArray = new Array(numRows);

		for (let i = 0; i < numRows; i++) {
		resultArray[i] = new Array(numCols);
		for (let j = 0; j < numCols; j++) {
		  resultArray[i][j] = array1[i][j] * array2[i][j];
		}
		}

		return resultArray;
	}
	
	static divide(matrixA, matrixB) {
		if (
		matrixA.length !== matrixB.length ||
		matrixA[0].length !== matrixB[0].length
		) {
		throw new Error('Matrix dimensions must be the same for division.');
		}

		const numRows = matrixA.length;
		const numCols = matrixA[0].length;
		const resultMatrix = [];

		for (let i = 0; i < numRows; i++) {
		resultMatrix.push([]);
		for (let j = 0; j < numCols; j++) {
		  resultMatrix[i][j] = matrixA[i][j] / matrixB[i][j];
		}
		}

		return resultMatrix;
	}

	static divideScalar(matrix, scalar) {
		const numRows = matrix.length;
		const numCols = matrix[0].length;
		const resultMatrix = [];

		for (let i = 0; i < numRows; i++) {
		resultMatrix.push([]);
		for (let j = 0; j < numCols; j++) {
		  resultMatrix[i][j] = matrix[i][j] / scalar;
		}
		}

		return resultMatrix;
	}

	static mean(matrix) {
		const numRows = matrix.length;
		const numCols = matrix[0].length;
		let sum = 0;

		for (let i = 0; i < numRows; i++) {
		for (let j = 0; j < numCols; j++) {
		  sum += matrix[i][j];
		}
		}

		return sum / (numRows * numCols);
	}

	static power(matrix, exponent) {
		const numRows = matrix.length;
		const numCols = matrix[0].length;
		const result = [];

		for (let i = 0; i < numRows; i++) {
		result.push([]);
		for (let j = 0; j < numCols; j++) {
		  result[i][j] = Math.pow(matrix[i][j], exponent);
		}
		}

		return result;
	}

	static size(matrix) {
		
		const numRows = matrix.length;
		const numCols = matrix[0].length;
		let sum = 0;

		for (let i = 0; i < numRows; i++) {
		for (let j = 0; j < numCols; j++) {
		  sum ++;
		}
		}
		
		return sum
	}
	
	static shape(matrix) {
		
		const numRows = matrix.length;
		const numCols = matrix[0].length;
		
		return [numRows, numCols];
		
	}
	
	static forEach(arr, callback) {
		
		const numRows = arr.length;
		const numCols = arr[0].length;
		const resultArray = new Array(numRows);
		
		for (let i = 0; i < numRows; i++) {
			resultArray[i] = new Array(numCols);
			for (let j = 0; j < numCols; j++) {
				resultArray[i][j] = callback(arr[i][j], arr);
			}
		}
		
		return resultArray;
		
	}

	static add(matrixA, matrixB) {
		
	  if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
		throw new Error('Matrices must have the same dimensions for addition.');
	  }

	  const numRows = matrixA.length;
	  const numCols = matrixA[0].length;
	  const result = [];

	  for (let i = 0; i < numRows; i++) {
		result.push([]);
		for (let j = 0; j < numCols; j++) {
		  result[i][j] = matrixA[i][j] + matrixB[i][j];
		}
	  }

	  return result;
	}

	static sub(matrixA, matrixB) {
	  if (matrixA.length !== matrixB.length || matrixA[0].length !== matrixB[0].length) {
		console.log(matrixA.length, matrixA[0].length, matrixB.length, matrixB[0].length)
		throw new Error('Matrices must have the same dimensions for subtraction.');
	  }

	  const numRows = matrixA.length;
	  const numCols = matrixA[0].length;

	  // Create a new matrix to store the result
	  const resultMatrix = new Array(numRows);

	  for (let i = 0; i < numRows; i++) {
		resultMatrix[i] = new Array(numCols);
		for (let j = 0; j < numCols; j++) {
		  resultMatrix[i][j] = matrixA[i][j] - matrixB[i][j];
		}
	  }

	  return resultMatrix;
	}
	
	static transpose(matrix) {
		
	  const numRows = matrix.length;
	  const numCols = matrix[0].length;

	  if (!numCols) {
		return [matrix]
	  }

	  // Create a new matrix with switched dimensions
	  const transposedMatrix = new Array(numCols);

	  for (let i = 0; i < numCols; i++) {
		transposedMatrix[i] = new Array(numRows);
		for (let j = 0; j < numRows; j++) {
		  transposedMatrix[i][j] = matrix[j][i];
		}
	  }

	  return transposedMatrix;
	}

	static generateStandardNormalRandom() {
	  let u = 0;
	  let v = 0;
	  while (u === 0) u = Math.random(); // Ensure u is not zero
	  while (v === 0) v = Math.random(); // Ensure v is not zero

	  const num = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
	  return num;
	}

	static random(arrs, elemsIn) {
	
	const a = [];
	for (let i = 0; i < arrs; i++) {
		const b = [];
		for (let j =  0; j < elemsIn; j++) {
			b.push(Matrix.generateStandardNormalRandom());
		}
		a.push(b);
	}
	return a;
}
}

class Layer {
	constructor () {
		this.type = "Layer";
		this.input = null;
		this.output = null;
	}
}

class Dense extends Layer {
	constructor (inputSize, outputSize, weights=null, biases=null){
		
		super();
		this.type = "Dense";
		this.weights = weights || Matrix.random(outputSize, inputSize);
		this.bias = biases || Matrix.random(outputSize, 1);		
		
	}
	
	forward (input) {
		
		this.input = input;
		return Matrix.add(Matrix.dot(this.weights, this.input ), this.bias );
		
	}
	
	backward (outputGradient, learningRate) {		
		
		const weightsGradient = Matrix.dot(outputGradient, Matrix.transpose(this.input));		
		this.weights = Matrix.sub(  this.weights,  Matrix.multiplyInt(weightsGradient, learningRate) );		
		this.bias = Matrix.sub( this.bias, Matrix.multiplyInt(outputGradient, learningRate) );		
		return Matrix.dot( Matrix.transpose(this.weights) , outputGradient )	
		
	}
}

class Activation extends Layer {
	constructor (activation, activationPrime) {
		
		super();
		this.type = "Activation";
		this.activation = activation;
		this.activationPrime = activationPrime;
		
	}
	
	forward (input) {
		
		this.input = input;
		return Matrix.forEach(this.input, this.activation);
		
	}
	
	backward (outputGradient, learningRate) {
		
		return Matrix.multiply(outputGradient, Matrix.forEach(this.input, this.activationPrime))
		
	}
	
}

function mse(ytrue, ypred) {
	return Matrix.mean(Matrix.power( Matrix.sub(ytrue, ypred), 2))
}

function msePrime(ytrue, ypred) {
	return Matrix.multiplyInt( Matrix.divideScalar ( Matrix.sub(ypred, ytrue), Matrix.size(ytrue)) , 2 )
}

function leakyReLU(x) {
  return x > 0 ? x : 0.01 * x;
}

function leakyReLUDerivative(x) {
  return x > 0 ? 1 : 0.01;
}

function max(x, matrix) {
	const numRows = matrix.length;
	const numCols = matrix[0].length;
	let total = 0;
	for (let i = 0; i < numRows; i++) {
	  for (let j = 0; j < numCols; j++) {
		total += matrix[i][j]; 
	  }
	}
	return x/total;
}

function maxPrime(x) {
	return x;
}

class Network {

	constructor (network=null, xtrain=null, ytrain=null) {
		this.network = network || [];
		this.xtrain = xtrain || [];
		this.ytrain = ytrain || [];
	}

	static toJSON (n, serialiseTraining=true) {
		
		const data = {
			network : []
		}
		
		if (serialiseTraining) {
			data.xtrain = n.xtrain,
			data.ytrain = n.ytrain
		}
			
		for (let i = 0; i < n.network.length; i++) {
			const lyr = n.network[i];
			const slyr = {};
			for (const key in lyr) {
				if (typeof lyr[key] === "function") {
					slyr[key] = {
						type : "function",
						name : lyr[key].name,
						value : lyr[key].toString().startsWith("function ") ? lyr[key].toString() : "function " + lyr[key].toString()
					}
				} else {
					slyr[key] = {
						type : (typeof lyr[key]),
						value : lyr[key]
					}
				}
			}
			data.network.push(slyr)
		}
		
		return data
		
	}
	
	toString() {
		return JSON.stringify(Network.toJSON);
	};
	
	static from(str) {
		return Network.fromJSON(str);
	};
	
	static fromJSON (data) {
		
		data = typeof data == "object" ? data : JSON.parse(data);
		
		const network = [];
		const xtrain = data.hasOwnProperty("xtrain") ? data.xtrain : null;
		const ytrain = data.hasOwnProperty("ytrain") ? data.ytrain : null;
		
		for (let i = 0; i < data.network.length; i++) {
			const slyr = data.network[i];
			let lyr = {};
			
			if (slyr.type.value == "Dense") {
				lyr = new Dense(1, 1);
			} else if (slyr.type.value == "Activation") {
				lyr = new Activation(() => {}, () => {});
			};
						
			for (const key in slyr) {
				if (slyr[key].type == "function") {
					const generatorFunction = new Function(`${slyr[key].value}\n\nreturn ${slyr[key].name}`);
					lyr[key] = generatorFunction();
				} else {
					lyr[key] = slyr[key].value;
				}
			}
			network.push(lyr)
		}
		
		return new Network(network, xtrain, ytrain)
		
	}
	
	static async evolve(requestedparams, initialpopulation, testingcallback, learningCallback=null) {
		
		const params = {
			generations : 10000,
			population : 20,
			survivors : 1,
			mutation : 0.1,
			mutationBySize : true,
		}
		
		for (const key in requestedparams) {
			params[key] = requestedparams[key];
		};
		
		const nets = [];
		let ids = [];
		
		function genid() {
			const id = Math.random()
			if (!ids.includes(id)) {
				return id
			} else { return genid() };
		}
		
		function giveid(net) {
			return {id : genid(), net : net};
		}

		function cleardeadids() {
			
			const newids = [];
			
			nets.forEach(net => newids.push(net.id));
			
			ids = newids;
			
		}


		function populate(survivors) {
			
			while (nets.length < params.population) {
				const survivor = survivors[nets.length % survivors.length];
				const offspring = Network.clone(survivor.net);//.clone();

				offspring.mutate(params.mutation, params.mutationBySize);
				
				const trackedNetwork = giveid(offspring);
				
				nets.push(trackedNetwork);
			}
			
		}

		async function sort() {
			
			const scores = {};
			const promises = [];
			
			nets.forEach((trackedNetwork) => {
				promises.push(new Promise(async (r) => {
					const newscore = await testingcallback(trackedNetwork.net);
					scores[trackedNetwork.id] = newscore;
					r();
				}));
			});
			
			for (let i = 0; i < promises.length; i++) {
				await promises[i];
			};
			
			nets.sort((a, b) => {return scores[b.id] - scores[a.id]});
			
			return scores;

		}
		
		function depopulate() {

			while (nets.length > params.survivors) {
				nets.pop();
			}; 			 
			
			cleardeadids();
			
		}
		
		initialpopulation.forEach((e) => {nets.push(giveid(e))})
		
		populate([...nets]);
		
		for (let generation = 0; generation < params.generations; generation ++) {
			
			const scores = await sort();
			if (learningCallback) {
				learningCallback(generation, scores, nets);
			};
			depopulate();
			populate([...nets]);
		
		}
		
		await sort();
		
		return nets[0].net;
		
	}
	
	static clone(net) {
		return Network.fromJSON(Network.toJSON(net, true));
	}
	
	clone() {
		return Network.fromJSON(Network.toJSON(this, true));
	}
	
	mutate(mutateAmount, divideBySize=true) {
		const mutateMultiplier = divideBySize ? mutateAmount / this.network.length : mutateAmount;
		for (let i = 0; i < this.network.length; i++) {
			const lyr = this.network[i];
			if (lyr.type == "Dense") {
				const shape = Matrix.shape(lyr.weights);
				const weightsGradient = Matrix.multiplyInt( Matrix.random(shape[0], shape[1]), mutateMultiplier );
				const biasGradient = Matrix.multiplyInt( Matrix.random(shape[0], 1) , mutateMultiplier );
				lyr.weights = Matrix.add(lyr.weights, weightsGradient);
				lyr.bias = Matrix.add(lyr.bias, biasGradient)
			}
		}
	}
	
	forward (input) {
		let output = input
		for (let i = 0; i < this.network.length; i++) {
			const lyr = this.network[i];
			output = lyr.forward(output);
		}
		return output;
	}
	
	backward (g, lr) {
		let gradient = g
		for (let i = this.network.length - 1; i >= 0; i--) {
			const lyr = this.network[i];
			gradient = lyr.backward(gradient, lr);
		}
	}
	
	linearRegress (x, y, learningRate, errorcb, errorprcb) {
		let e = 0;
		
		const output = this.forward(x);
		e += errorcb(y, output);
		
		this.backward(errorprcb(y, output), learningRate);
		e /= this.xtrain.length;
		
		return e;
	}
	
	train (epochs, learningRate, errorcb=null, errorprcb=null, cb=null) {
		
		errorcb = errorcb || mse;
		errorprcb = errorprcb || msePrime;
		
		for (let e = 0; e < epochs; e++) {
			
			let error = 0;
			
			for (let i = 0; i < this.xtrain.length; i ++) {
				
				const x = this.xtrain[i];
				const y = this.ytrain[i];
				
				error += this.linearRegress(x, y, learningRate, errorcb, errorprcb) 
			
			}

			if (cb) {
				cb(e, error);
			}
			
		}
		
	}

}

return {
	meanSquaredError : mse,
	meanSquaredErrorPrime : msePrime,
	mse : mse,
	msePrime : msePrime,
	Network : Network,
	Matrix : Matrix,
	Layer : Layer,
	Dense : Dense,
	Activation : Activation,
	max : max,
	maxPrime : maxPrime,
	leakyReLU : leakyReLU,
	leakyReLUPrime : leakyReLUDerivative,
}
