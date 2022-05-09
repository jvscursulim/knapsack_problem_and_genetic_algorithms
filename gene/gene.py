
class Gene:
    """Gene class"""
    def __init__(self, weight: int, points: int) -> None:
        """Creates a gene object.

        Args:
            weight (int): The weight of the gene.
            points (int): The points of the gene.
        """
        self.weight = weight
        self.points = points