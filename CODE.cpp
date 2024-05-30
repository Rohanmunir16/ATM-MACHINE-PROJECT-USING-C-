#include <iostream>   //liabrary for cin,cout
#include <fstream>   //liabrary for file handling
using namespace std;

//declaration of functions
double withdrawal(int& balance);
double deposit(int& balance);
double fundtransfer(int& balance);
double PIN(int& pin, int& your_pin);
void saveData(int balance, int your_pin);
void loadData(int& balance, int& your_pin);

int main() {
    int pin;
    char option;
    int balance; //variable to store balance
    int your_pin;//variable to store PIN


    loadData(balance, your_pin);

    do {
        //loop will continue to run if the password is correct
        while (true) {
            cout << "--------Welcome to Habib Bank Limited-------" << endl;
            cout << endl;
            cout << "Please Insert your Card" << endl;
            cout << endl;

            cout << "ENTER YOUR PIN" << endl;
            cin >> pin;
            //if the password is correct
            if (pin == your_pin) {
                cout << "\t\t\t\t\t*Your transaction has been processed*\t\t\t\t\t" << endl;
                cout << endl;
                break;
            }
            //if the password is not valid
            else {
                cout << "INVALID PIN. Please Try Again" << endl;
            }
        }
        //user will select the mode of transaction
        cout << "Please Select Your Mode Of Transaction" << endl;
        cout << endl;
        cout << "A. Withdrawal" << endl;
        cout << "B. Deposit" << endl;
        cout << "C. Fund Transfer" << endl;
        cout << "D. Balance Inquiry" << endl;
        cout << "E. Change PIN" << endl;
        cout << "F. EXIT" << endl;
        cout << endl;
        cin >> option;

        // Execute the corresponding function based on user input
        switch (option) {
        case 'A':
            withdrawal(balance); //call withdrawal function
            break;
        case 'B':
            deposit(balance);  //call deposit function
            break;
        case 'C':
            fundtransfer(balance);  //call fundtransfer function
            break;
        case 'D':
            cout << "Your Current Balance is " << balance << endl; //current balance will be displayed
            break;
        case 'E':
            PIN(pin, your_pin); //call PIN function
            break;
        case 'F':
            cout << "Thank you for using this ATM" << endl;
            break;
        default:
            cout << "Invalid Option. Please try again." << endl;
        }
        cout << endl;
        cout << "\t\t\t\t\t\tTHANK YOU!\t\t\t\t\t\t\t\t\t" << endl;
        cout << "\t\t\t\t\t*Please take your card*\t\t\t\t\t" << endl;
        cout << endl;

        // Save balance and PIN to file after each transaction
        saveData(balance, your_pin);

    } while (option != 'F');  // loop will end if the user enters F

    return 0;
}

double withdrawal(int& balance) {
    int amount_1;    // variable to store the amount for the transaction
    cout << "Please enter the amount for the transaction" << endl;
    cin >> amount_1;
    cout << endl;
    //if condition is valid
    if (amount_1 > 0 && amount_1 <= balance) {
        cout << "TRANSACTION SUCCESSFUL" << endl;
        cout << "Please Collect Your Cash" << endl;
        balance -= amount_1;        //deduct the amount from the balance
        cout << "Your Remaining Balance is " << balance << endl;
    }
    //if condition is not valid
    else {
        cout << "INSUFFICIENT FUNDS" << endl;
    }
    return balance; //Return the updated balance
}

double deposit(int& balance) {
    int amount_2;   // variable to store the amount for the transaction
    cout << "Enter the amount you want to deposit" << endl;
    cin >> amount_2;
    cout << endl;
    //if condition is valid
    if (amount_2 > 0) {
        cout << "DEPOSIT SUCCESSFUL" << endl;
        balance += amount_2;  //add the amount to the balance
        cout << "Your new balance is " << balance << endl;
    }
    //if condition is not valid
    else {
        cout << "INVALID DEPOSIT. Please Try Again" << endl;
    }
    return balance;  //Return the updated balance
}

double fundtransfer(int& balance) {
    int amount_3;    // variable to store the amount for the transaction
    cout << "Enter the amount you want to transfer" << endl;
    cin >> amount_3;
    cout << endl;
    //if condition is valid
    if (amount_3 > 0 && amount_3 <= balance) {
        cout << "Your amount has been transferred" << endl;
        balance -= amount_3; //deduct the amount from the balance
        cout << "Your remaining balance is " << balance << endl;
    }
    //if condition is not valid
    else {
        cout << "INVALID TRANSFER" << endl;
    }
    return balance;   //Return the updated balance
}

double PIN(int& pin, int& your_pin) {
    int old_pin, new_pin, confirm_pin;
    cout << "Enter Your Previous PIN" << endl;
    cin >> old_pin;

    // Check if the old PIN entered matches the current PIN
    if (old_pin == your_pin) {
        cout << "Enter your new PIN" << endl;
        cin >> new_pin;
        cout << "Confirm your new PIN" << endl;
        cin >> confirm_pin;
        // Check if the new PIN and confirmed new PIN match
        if (new_pin == confirm_pin) {
            your_pin = new_pin;      // Update the PIN to the new PIN
            cout << "Your PIN has been changed successfully to " << new_pin << endl;
        }
        else {
            cout << "New PIN and confirm PIN do not match. Please try again." << endl;
        }
    }

    // Inform the user that the old PIN entered is incorrect

    else {
        cout << "OLD PIN INCORRECT" << endl;
    }
    return your_pin;  //return the updated PIN
}
//Function to save balance and file
void saveData(int balance, int your_pin) {
    ofstream outfile("account_data.txt");
    if (outfile.is_open()) {
        outfile << balance << endl;//write balance to the file
        outfile << your_pin << endl; // write PIN to the file
        outfile.close();
    }
    else {
        cout << "Unable to open file for writing." << endl;
    }
}
//Function to load balance and file
void loadData(int& balance, int& your_pin) {
    ifstream infile("account_data.txt");
    if (infile.is_open()) {
        infile >> balance; // Read balance to the file
        infile >> your_pin; //Read balance to the file
        infile.close();
    }
    else {

        balance = 50000;
        your_pin = 1926;
    }
}
