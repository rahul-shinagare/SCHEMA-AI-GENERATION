import json
from torch.utils.data import Dataset


class SchemaDataset(Dataset):
    """
    Synthetic Dataset for database schema generation
    """

    def __init__(self):
        # Example schema generation data
        self.data = [
            {
                "prompt": "Design a database schema for a book store.",
                "schema": {
                    "tables": [
                        {
                            "name": "books",
                            "fields": [
                                {"name": "book_id", "type": "number"},
                                {"name": "title", "type": "text"},
                                {"name": "author", "type": "text"},
                                {"name": "price", "type": "number"}
                            ]
                        },
                        {
                            "name": "customers",
                            "fields": [
                                {"name": "customer_id", "type": "number"},
                                {"name": "name", "type": "text"},
                                {"name": "email", "type": "email"}
                            ]
                        }
                    ]
                }
            },
            {
                "prompt": "Create a schema for an online food delivery system.",
                "schema": {
                    "tables": [
                        {
                            "name": "orders",
                            "fields": [
                                {"name": "order_id", "type": "number"},
                                {"name": "customer_id", "type": "number"},
                                {"name": "restaurant_id", "type": "number"},
                                {"name": "order_date", "type": "date"}
                            ]
                        },
                        {
                            "name": "restaurants",
                            "fields": [
                                {"name": "restaurant_id", "type": "number"},
                                {"name": "name", "type": "text"},
                                {"name": "location", "type": "text"}
                            ]
                        },
                        {
                            "name": "customers",
                            "fields": [
                                {"name": "customer_id", "type": "number"},
                                {"name": "name", "type": "text"},
                                {"name": "phone", "type": "text"}
                            ]
                        }
                    ]
                }
            },
            {
                "prompt": "Design a database schema for a university system.",
                "schema": {
                    "tables": [
                        {
                            "name": "students",
                            "fields": [
                                {"name": "student_id", "type": "number"},
                                {"name": "name", "type": "text"},
                                {"name": "email", "type": "email"}
                            ]
                        },
                        {
                            "name": "courses",
                            "fields": [
                                {"name": "course_id", "type": "number"},
                                {"name": "course_name", "type": "text"},
                                {"name": "credits", "type": "number"}
                            ]
                        },
                        {
                            "name": "enrollments",
                            "fields": [
                                {"name": "enrollment_id", "type": "number"},
                                {"name": "student_id", "type": "number"},
                                {"name": "course_id", "type": "number"},
                                {"name": "semester", "type": "text"}
                            ]
                        }
                    ]
                }
            }
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Return tokenized prompt and schema pair
        """
        return {
            "prompt": self.data[idx]["prompt"],
            "schema": json.dumps(self.data[idx]["schema"])  # Ensure schema is string-encoded
        }
