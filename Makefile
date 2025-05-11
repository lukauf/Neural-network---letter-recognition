PYTHON=python3

install:
	$(PYTHON) -m pip install -r requirements.txt

letters:
	$(PYTHON) NNW_Letters.py

letters-cross-validation:
	$(PYTHON) NNW_Letters_Cross_Validation.py

letters-early-stopping:
	$(PYTHON) NNW_Letters_Early_Stopping.py

logic-gates-AND:
	$(PYTHON) NNW_Logic_Gates.py AND

logic-gates-OR:
	$(PYTHON) NNW_Logic_Gates.py OR

logic-gates-XOR:
	$(PYTHON) NNW_Logic_Gates.py XOR