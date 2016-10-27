// vector::push_back
#include <iostream>
#include <vector>

int main ()
{
  std::vector<int> myvector;
  int myint;

  std::cout << "Please enter some integers (enter 0 to end):\n";

  for(int i=0;i<3;i++) {
    std::cin >> myint;
    myvector.push_back (myint);
    //std::cout << "myvector["<<i<<"]: " << myvector[i] << " \n";
  } 
  for(int i=0;i<myvector.size(); i++){
  std::cout << "myvector["<<i<<"]: " << myvector[i] << " \n";
}
  return 0;
}