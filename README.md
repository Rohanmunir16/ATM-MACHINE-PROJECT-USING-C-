# ATM Management System

A simple console-based **ATM Management System** developed in **C++** that simulates basic banking operations such as withdrawals, deposits, balance inquiry, fund transfers, and PIN management with file handling support for data persistence. 

---

## Features

* Secure PIN Authentication
* Cash Withdrawal
* Cash Deposit
* Fund Transfer
* Balance Inquiry
* Change ATM PIN
* Data Persistence using File Handling
* Simple Console Interface

---

## Technologies Used

* **C++**
* **File Handling (fstream)**
* **Object-Oriented & Modular Programming Concepts**

---

## Project Structure

```bash
ATM.cpp                # Main source code
account_data.txt       # Stores balance and PIN data
```

---

## How It Works

1. User inserts card (simulation).
2. System asks for PIN verification.
3. After successful login, user can:

   * Withdraw money
   * Deposit money
   * Transfer funds
   * Check balance
   * Change PIN
4. Updated balance and PIN are automatically saved in a file.

---

## Functions Used

| Function         | Purpose                      |
| ---------------- | ---------------------------- |
| `withdrawal()`   | Withdraw money from account  |
| `deposit()`      | Deposit money into account   |
| `fundtransfer()` | Transfer funds               |
| `PIN()`          | Change account PIN           |
| `saveData()`     | Save balance & PIN into file |
| `loadData()`     | Load saved balance & PIN     |

---

## Default Account Details

If no saved file exists, the program initializes with:

```txt
Default Balance: 50000
Default PIN: 1926
```

---

## How to Run

### Using g++

```bash
g++ ATM.cpp -o atm
./atm
```

### Using Visual Studio

1. Open the project in Visual Studio
2. Build the project
3. Run the program

---

## Sample Menu

```txt
A. Withdrawal
B. Deposit
C. Fund Transfer
D. Balance Inquiry
E. Change PIN
F. Exit
```

---

## Concepts Demonstrated

* Functions
* Loops
* Conditional Statements
* File Handling
* User Input Validation
* Modular Programming

---

## Future Improvements

* Add multiple user accounts
* Encrypt PIN for better security
* Add transaction history
* GUI-based ATM interface
* Database integration
* Admin panel

---

## Author

Developed by **Rohan Munir**

---

## License

This project is created for educational and learning purposes.

---

## Source File

Main implementation file

