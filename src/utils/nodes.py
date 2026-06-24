"""Tree node data structures shared by SPLIT, solver, and builder."""


class SPLITLeaf:
    """Leaf node for SPLIT trees - stores a class prediction and loss."""

    __slots__ = ("prediction", "loss")

    def __init__(self, prediction: int, loss: float):
        self.prediction = prediction
        self.loss = loss

    def __str__(self) -> str:
        return (
            "{ prediction: "
            + str(self.prediction)
            + ", loss: "
            + str(self.loss)
            + " }"
        )

    def __repr__(self) -> str:
        return self.__str__()


class SPLITNode:
    """Internal node for SPLIT trees - stores a binary feature index and
    pointers to left (feature == True) and right (feature == False) children."""

    __slots__ = ("feature", "left_child", "right_child")

    def __init__(self, feature: int, left_child, right_child):
        self.feature = feature
        self.left_child = left_child
        self.right_child = right_child

    def __str__(self) -> str:
        return (
            "{ feature: "
            + str(self.feature)
            + " [ left child: "
            + str(self.left_child)
            + ", right child: "
            + str(self.right_child)
            + "] }"
        )

    def __repr__(self) -> str:
        return self.__str__()
