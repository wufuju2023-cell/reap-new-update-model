import ReapRuntime
def ClosedFixture.trueClaim : Prop := forall n : Nat, n = n
def ClosedFixture.falseClaim : Prop := forall n : Nat, n = 0
def ClosedFixture.parameterized (n : Nat) : Prop := n = n
def ClosedFixture.boolTarget : Bool := true
