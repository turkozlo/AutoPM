

import pandas as pd


def generate_test_excel(file_path="processes.xlsx"):
    data = {
        "Process ID": [
            "P001", "P002", "P003", "P004", "P005",
            "P101", "P102", "P103", "P104", "P105"
        ],
        "Process Name": [
            "Order to Cash",
            "Procure to Pay",
            "Hire to Retire",
            "Record to Report",
            "Issue to Resolution",
            "Loan Application",
            "Insurance Claim Processing",
            "Customer Onboarding",
            "Technical Support Workflow",
            "Inventory Management"
        ],
        "Description": [
            "Standard process for handling customer orders from placement to payment.",
            "Process for acquiring goods and services and paying for them.",
            "Lifecycle of an employee from recruitment to retirement.",
            "Financial reporting process from recording transactions to final reports.",
            "Handling customer issues from initial report to final resolution.",
            "End-to-end workflow for processing personal and mortgage loan applications.",
            "Workflow for evaluating and paying out insurance claims.",
            "Steps to register and verify a new customer in the system.",
            "Tiered support process for resolving technical customer queries.",
            "Monitoring and managing stock levels and warehouse operations."
        ]
    }

    df = pd.DataFrame(data)
    df.to_excel(file_path, index=False)
    print(f"Test file generated: {file_path}")


if __name__ == "__main__":
    generate_test_excel()
