PYTHON=python3
LETTER = NNW_Letters
GATES = NNW_Logic_Gates

install:
	$(PYTHON) -m pip install -r requirements.txt

letters:
	$(PYTHON) $(LETTER).py

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