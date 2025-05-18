PYTHON=python3
LETTER = NNW_Letters
GATES = NNW_Logic_Gates
GRID = NNW_Grid_Search_Hyperparams
BENCHMARK = NNW_Benchmark_Train_MLP

install:
	$(PYTHON) -m pip install -r requirements.txt

grid-search:
	$(PYTHON) $(GRID).py

benchmark:
	$(PYTHON) $(BENCHMARK).py

letters:
	$(PYTHON) $(LETTER).py a

letters-test:
	$(PYTHON) $(LETTER).py yes

letters-cross-validation:
	$(PYTHON) $(LETTER)_Cross_Validation.py

letters-early-stopping:
	$(PYTHON) $(LETTER)_Early_Stopping.py

logic-gates-AND:
	$(PYTHON) $(GATES).py AND

logic-gates-OR:
	$(PYTHON) $(GATES).py OR

logic-gates-XOR:
	$(PYTHON) $(GATES).py XOR
