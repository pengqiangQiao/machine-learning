"""
Python基础语法 - 面向Java程序员
Python Basics for Java Programmers

这个文件涵盖了Python的基础语法，并对照Java实现
This file covers Python basics with Java equivalents
"""


# ============================================================
# 1. 变量和数据类型
# Variables and Data Types
# ============================================================

def variables_and_types():
    """
    变量和数据类型
    
    Java对应：
    public void variablesAndTypes() {
        // Java需要声明类型
        int number = 10;
        double decimal = 3.14;
        String text = "Hello";
        boolean flag = true;
        
        // Java的类型转换
        String numStr = String.valueOf(number);
        int parsedNum = Integer.parseInt("123");
    }
    """
    print("=" * 50)
    print("1. 变量和数据类型")
    print("=" * 50)

    # Python不需要声明类型（动态类型）
    # Java: int number = 10;
    number = 10

    # Java: double decimal = 3.14;
    decimal = 3.14

    # Java: String text = "Hello";
    text = "Hello"

    # Java: boolean flag = true;
    flag = True  # 注意：Python用True/False，Java用true/false

    # Java: Object obj = null;
    obj = None  # Python用None表示空，Java用null

    print(f"整数: {number}, 类型: {type(number)}")
    print(f"浮点数: {decimal}, 类型: {type(decimal)}")
    print(f"字符串: {text}, 类型: {type(text)}")
    print(f"布尔值: {flag}, 类型: {type(flag)}")

    # 类型转换
    # Java: String numStr = String.valueOf(number);
    num_str = str(number)

    # Java: int parsedNum = Integer.parseInt("123");
    parsed_num = int("123")

    # Java: double parsedDouble = Double.parseDouble("3.14");
    parsed_double = float("3.14")

    print(f"\n类型转换: {num_str}, {parsed_num}, {parsed_double}")


# ============================================================
# 2. 字符串操作
# String Operations
# ============================================================

def string_operations():
    """
    字符串操作
    
    Java对应：
    public void stringOperations() {
        String str = "Hello World";
        
        // 字符串拼接
        String result = str + "!";
        String formatted = String.format("Hello %s", "Python");
        
        // 字符串方法
        String upper = str.toUpperCase();
        String lower = str.toLowerCase();
        boolean contains = str.contains("World");
        String replaced = str.replace("World", "Java");
        String[] parts = str.split(" ");
        String trimmed = str.trim();
        
        // 字符串长度
        int length = str.length();
        
        // 字符串切片（Java用substring）
        String sub = str.substring(0, 5);
    }
    """
    print("\n" + "=" * 50)
    print("2. 字符串操作")
    print("=" * 50)

    # Java: String str = "Hello World";
    text = "Hello World"

    # 字符串拼接
    # Java: String result = str + "!";
    result = text + "!"

    # 格式化字符串（Python有多种方式）
    # Java: String formatted = String.format("Hello %s", "Python");
    formatted1 = "Hello %s" % "Python"  # 旧式
    formatted2 = "Hello {}".format("Python")  # format方法
    formatted3 = f"Hello {'Python'}"  # f-string（推荐）

    print(f"拼接: {result}")
    print(f"格式化: {formatted3}")

    # 字符串方法
    # Java: String upper = str.toUpperCase();
    upper = text.upper()

    # Java: String lower = str.toLowerCase();
    lower = text.lower()

    # Java: boolean contains = str.contains("World");
    contains = "World" in text

    # Java: String replaced = str.replace("World", "Java");
    replaced = text.replace("World", "Java")

    # Java: String[] parts = str.split(" ");
    parts = text.split(" ")

    # Java: String trimmed = "  hello  ".trim();
    trimmed = "  hello  ".strip()

    print(f"大写: {upper}")
    print(f"小写: {lower}")
    print(f"包含'World': {contains}")
    print(f"替换: {replaced}")
    print(f"分割: {parts}")
    print(f"去空格: '{trimmed}'")

    # 字符串长度
    # Java: int length = str.length();
    length = len(text)
    print(f"长度: {length}")

    # 字符串切片（Python特有，非常强大）
    # Java: String sub = str.substring(0, 5);
    sub = text[0:5]  # 获取索引0到4的字符
    print(f"切片[0:5]: {sub}")
    print(f"切片[6:]: {text[6:]}")  # 从索引6到结尾
    print(f"切片[:5]: {text[:5]}")  # 从开始到索引4
    print(f"反转: {text[::-1]}")  # 反转字符串


# ============================================================
# 3. 列表（List）- 对应Java的ArrayList
# Lists - Java's ArrayList equivalent
# ============================================================

def list_operations():
    """
    列表操作
    
    Java对应：
    public void listOperations() {
        // Java使用ArrayList
        List<Integer> numbers = new ArrayList<>();
        
        // 添加元素
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        
        // 访问元素
        int first = numbers.get(0);
        
        // 修改元素
        numbers.set(0, 10);
        
        // 删除元素
        numbers.remove(0);  // 按索引删除
        numbers.remove(Integer.valueOf(2));  // 按值删除
        
        // 列表大小
        int size = numbers.size();
        
        // 检查是否包含
        boolean contains = numbers.contains(3);
        
        // 清空列表
        numbers.clear();
        
        // 遍历列表
        for (int num : numbers) {
            System.out.println(num);
        }
        
        // 列表排序
        Collections.sort(numbers);
        
        // 列表反转
        Collections.reverse(numbers);
    }
    """
    print("\n" + "=" * 50)
    print("3. 列表操作（List）")
    print("=" * 50)

    # 创建列表
    # Java: List<Integer> numbers = new ArrayList<>();
    numbers = []  # 空列表
    numbers2 = [1, 2, 3, 4, 5]  # 带初始值的列表

    # 添加元素
    # Java: numbers.add(1);
    numbers.append(1)
    numbers.append(2)
    numbers.append(3)
    print(f"添加后: {numbers}")

    # 在指定位置插入
    # Java: numbers.add(0, 10);
    numbers.insert(0, 10)
    print(f"插入后: {numbers}")

    # 访问元素
    # Java: int first = numbers.get(0);
    first = numbers[0]
    last = numbers[-1]  # Python可以用负索引，-1表示最后一个
    print(f"第一个: {first}, 最后一个: {last}")

    # 修改元素
    # Java: numbers.set(0, 100);
    numbers[0] = 100
    print(f"修改后: {numbers}")

    # 删除元素
    # Java: numbers.remove(0);  // 按索引
    del numbers[0]
    print(f"删除索引0后: {numbers}")

    # Java: numbers.remove(Integer.valueOf(2));  // 按值
    numbers.remove(2)  # 删除第一个值为2的元素
    print(f"删除值2后: {numbers}")

    # 列表大小
    # Java: int size = numbers.size();
    size = len(numbers)
    print(f"列表大小: {size}")

    # 检查是否包含
    # Java: boolean contains = numbers.contains(3);
    contains = 3 in numbers
    print(f"包含3: {contains}")

    # 列表拼接
    # Java: numbers.addAll(Arrays.asList(6, 7, 8));
    numbers.extend([6, 7, 8])
    print(f"拼接后: {numbers}")

    # 列表切片
    # Java: List<Integer> subList = numbers.subList(0, 3);
    sub_list = numbers[0:3]
    print(f"切片[0:3]: {sub_list}")

    # 列表排序
    # Java: Collections.sort(numbers);
    numbers.sort()
    print(f"排序后: {numbers}")

    # 列表反转
    # Java: Collections.reverse(numbers);
    numbers.reverse()
    print(f"反转后: {numbers}")

    # 清空列表
    # Java: numbers.clear();
    numbers_copy = numbers.copy()
    numbers_copy.clear()
    print(f"清空后: {numbers_copy}")


# ============================================================
# 4. 字典（Dictionary）- 对应Java的HashMap
# Dictionaries - Java's HashMap equivalent
# ============================================================

def dictionary_operations():
    """
    字典操作
    
    Java对应：
    public void dictionaryOperations() {
        // Java使用HashMap
        Map<String, Integer> ages = new HashMap<>();
        
        // 添加键值对
        ages.put("Alice", 25);
        ages.put("Bob", 30);
        ages.put("Charlie", 35);
        
        // 访问值
        int aliceAge = ages.get("Alice");
        
        // 检查键是否存在
        boolean hasKey = ages.containsKey("Alice");
        
        // 删除键值对
        ages.remove("Bob");
        
        // 获取所有键
        Set<String> keys = ages.keySet();
        
        // 获取所有值
        Collection<Integer> values = ages.values();
        
        // 遍历字典
        for (Map.Entry<String, Integer> entry : ages.entrySet()) {
            System.out.println(entry.getKey() + ": " + entry.getValue());
        }
        
        // 字典大小
        int size = ages.size();
        
        // 清空字典
        ages.clear();
    }
    """
    print("\n" + "=" * 50)
    print("4. 字典操作（Dictionary）")
    print("=" * 50)

    # 创建字典
    # Java: Map<String, Integer> ages = new HashMap<>();
    ages = {}  # 空字典
    ages2 = {"Alice": 25, "Bob": 30, "Charlie": 35}  # 带初始值

    # 添加键值对
    # Java: ages.put("Alice", 25);
    ages["Alice"] = 25
    ages["Bob"] = 30
    ages["Charlie"] = 35
    print(f"字典: {ages}")

    # 访问值
    # Java: int aliceAge = ages.get("Alice");
    alice_age = ages["Alice"]
    # 安全访问（如果键不存在返回默认值）
    # Java: int age = ages.getOrDefault("David", 0);
    david_age = ages.get("David", 0)
    print(f"Alice的年龄: {alice_age}")
    print(f"David的年龄（默认）: {david_age}")

    # 检查键是否存在
    # Java: boolean hasKey = ages.containsKey("Alice");
    has_alice = "Alice" in ages
    print(f"包含Alice: {has_alice}")

    # 修改值
    # Java: ages.put("Alice", 26);
    ages["Alice"] = 26
    print(f"修改后: {ages}")

    # 删除键值对
    # Java: ages.remove("Bob");
    del ages["Bob"]
    print(f"删除Bob后: {ages}")

    # 获取所有键
    # Java: Set<String> keys = ages.keySet();
    keys = ages.keys()
    print(f"所有键: {list(keys)}")

    # 获取所有值
    # Java: Collection<Integer> values = ages.values();
    values = ages.values()
    print(f"所有值: {list(values)}")

    # 获取所有键值对
    # Java: Set<Map.Entry<String, Integer>> entries = ages.entrySet();
    items = ages.items()
    print(f"所有键值对: {list(items)}")

    # 字典大小
    # Java: int size = ages.size();
    size = len(ages)
    print(f"字典大小: {size}")


# ============================================================
# 5. 集合（Set）- 对应Java的HashSet
# Sets - Java's HashSet equivalent
# ============================================================

def set_operations():
    """
    集合操作
    
    Java对应：
    public void setOperations() {
        // Java使用HashSet
        Set<Integer> numbers = new HashSet<>();
        
        // 添加元素
        numbers.add(1);
        numbers.add(2);
        numbers.add(3);
        numbers.add(2);  // 重复元素不会被添加
        
        // 删除元素
        numbers.remove(2);
        
        // 检查是否包含
        boolean contains = numbers.contains(1);
        
        // 集合大小
        int size = numbers.size();
        
        // 集合运算
        Set<Integer> set1 = new HashSet<>(Arrays.asList(1, 2, 3));
        Set<Integer> set2 = new HashSet<>(Arrays.asList(3, 4, 5));
        
        // 并集
        Set<Integer> union = new HashSet<>(set1);
        union.addAll(set2);
        
        // 交集
        Set<Integer> intersection = new HashSet<>(set1);
        intersection.retainAll(set2);
        
        // 差集
        Set<Integer> difference = new HashSet<>(set1);
        difference.removeAll(set2);
    }
    """
    print("\n" + "=" * 50)
    print("5. 集合操作（Set）")
    print("=" * 50)

    # 创建集合（自动去重）
    # Java: Set<Integer> numbers = new HashSet<>();
    numbers = set()  # 空集合
    numbers2 = {1, 2, 3, 4, 5}  # 带初始值

    # 添加元素
    # Java: numbers.add(1);
    numbers.add(1)
    numbers.add(2)
    numbers.add(3)
    numbers.add(2)  # 重复元素不会被添加
    print(f"集合: {numbers}")

    # 删除元素
    # Java: numbers.remove(2);
    numbers.remove(2)
    print(f"删除2后: {numbers}")

    # 安全删除（如果元素不存在不会报错）
    # Java: numbers.remove(10);  // 如果不存在会返回false
    numbers.discard(10)

    # 检查是否包含
    # Java: boolean contains = numbers.contains(1);
    contains = 1 in numbers
    print(f"包含1: {contains}")

    # 集合大小
    # Java: int size = numbers.size();
    size = len(numbers)
    print(f"集合大小: {size}")

    # 集合运算
    set1 = {1, 2, 3, 4}
    set2 = {3, 4, 5, 6}

    # 并集
    # Java: Set<Integer> union = new HashSet<>(set1); union.addAll(set2);
    union = set1 | set2  # 或 set1.union(set2)
    print(f"并集: {union}")

    # 交集
    # Java: Set<Integer> intersection = new HashSet<>(set1); intersection.retainAll(set2);
    intersection = set1 & set2  # 或 set1.intersection(set2)
    print(f"交集: {intersection}")

    # 差集
    # Java: Set<Integer> difference = new HashSet<>(set1); difference.removeAll(set2);
    difference = set1 - set2  # 或 set1.difference(set2)
    print(f"差集: {difference}")

    # 对称差集（在set1或set2中，但不在两者交集中）
    # Java: 需要手动实现
    symmetric_diff = set1 ^ set2  # 或 set1.symmetric_difference(set2)
    print(f"对称差集: {symmetric_diff}")


# ============================================================
# 6. 元组（Tuple）- 对应Java的不可变列表
# Tuples - Java's immutable list equivalent
# ============================================================

def tuple_operations():
    """
    元组操作（不可变列表）
    
    Java对应：
    public void tupleOperations() {
        // Java没有内置的元组，可以使用不可变列表
        List<Integer> tuple = Collections.unmodifiableList(
            Arrays.asList(1, 2, 3, 4, 5)
        );
        
        // 访问元素
        int first = tuple.get(0);
        
        // 元组大小
        int size = tuple.size();
        
        // 检查是否包含
        boolean contains = tuple.contains(3);
        
        // 注意：不能修改元组
        // tuple.set(0, 10);  // 会抛出UnsupportedOperationException
    }
    """
    print("\n" + "=" * 50)
    print("6. 元组操作（Tuple）")
    print("=" * 50)

    # 创建元组（不可变）
    # Java: List<Integer> tuple = Collections.unmodifiableList(Arrays.asList(1, 2, 3));
    numbers = (1, 2, 3, 4, 5)
    single = (1,)  # 单元素元组需要逗号

    print(f"元组: {numbers}")

    # 访问元素
    # Java: int first = tuple.get(0);
    first = numbers[0]
    last = numbers[-1]
    print(f"第一个: {first}, 最后一个: {last}")

    # 元组切片
    sub_tuple = numbers[1:4]
    print(f"切片[1:4]: {sub_tuple}")

    # 元组大小
    # Java: int size = tuple.size();
    size = len(numbers)
    print(f"元组大小: {size}")

    # 检查是否包含
    # Java: boolean contains = tuple.contains(3);
    contains = 3 in numbers
    print(f"包含3: {contains}")

    # 元组解包
    # Java: 需要手动赋值
    a, b, c, d, e = numbers
    print(f"解包: a={a}, b={b}, c={c}, d={d}, e={e}")

    # 元组拼接
    # Java: 需要创建新的列表
    tuple1 = (1, 2, 3)
    tuple2 = (4, 5, 6)
    combined = tuple1 + tuple2
    print(f"拼接: {combined}")

    # 注意：元组不可修改
    # numbers[0] = 10  # 这会报错！TypeError


# ============================================================
# 7. 控制流 - if/else
# Control Flow - if/else
# ============================================================

def control_flow_if():
    """
    条件语句
    
    Java对应：
    public void controlFlowIf() {
        int x = 10;
        
        // if语句
        if (x > 0) {
            System.out.println("正数");
        } else if (x < 0) {
            System.out.println("负数");
        } else {
            System.out.println("零");
        }
        
        // 三元运算符
        String result = (x > 0) ? "正数" : "非正数";
        
        // 逻辑运算符
        if (x > 0 && x < 100) {
            System.out.println("在范围内");
        }
        
        if (x < 0 || x > 100) {
            System.out.println("在范围外");
        }
        
        if (!(x == 0)) {
            System.out.println("不是零");
        }
    }
    """
    print("\n" + "=" * 50)
    print("7. 条件语句（if/else）")
    print("=" * 50)

    x = 10

    # if语句（注意Python用缩进表示代码块，不用大括号）
    # Java: if (x > 0) { ... }
    if x > 0:
        print("x是正数")
    elif x < 0:  # Java: else if
        print("x是负数")
    else:
        print("x是零")

    # 三元运算符
    # Java: String result = (x > 0) ? "正数" : "非正数";
    result = "正数" if x > 0 else "非正数"
    print(f"三元运算符: {result}")

    # 逻辑运算符
    # Java: if (x > 0 && x < 100)
    if x > 0 and x < 100:  # Python用and，Java用&&
        print("x在0到100之间")

    # Java: if (x < 0 || x > 100)
    if x < 0 or x > 100:  # Python用or，Java用||
        print("x不在0到100之间")

    # Java: if (!(x == 0))
    if not (x == 0):  # Python用not，Java用!
        print("x不是零")

    # 检查None
    # Java: if (obj != null)
    obj = None
    if obj is None:  # Python用is None，不用== None
        print("obj是None")

    # 检查空列表/字符串
    # Java: if (list.isEmpty())
    empty_list = []
    if not empty_list:  # Python中空列表/字符串/0都是False
        print("列表为空")


# ============================================================
# 8. 循环 - for和while
# Loops - for and while
# ============================================================

def loops():
    """
    循环语句
    
    Java对应：
    public void loops() {
        // for循环
        for (int i = 0; i < 5; i++) {
            System.out.println(i);
        }
        
        // 增强for循环（foreach）
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        for (int num : numbers) {
            System.out.println(num);
        }
        
        // while循环
        int i = 0;
        while (i < 5) {
            System.out.println(i);
            i++;
        }
        
        // do-while循环
        int j = 0;
        do {
            System.out.println(j);
            j++;
        } while (j < 5);
        
        // break和continue
        for (int k = 0; k < 10; k++) {
            if (k == 5) {
                break;  // 跳出循环
            }
            if (k % 2 == 0) {
                continue;  // 跳过本次循环
            }
            System.out.println(k);
        }
    }
    """
    print("\n" + "=" * 50)
    print("8. 循环语句（for/while）")
    print("=" * 50)

    # for循环 - 遍历范围
    # Java: for (int i = 0; i < 5; i++)
    print("for循环 - range(5):")
    for i in range(5):  # range(5)生成0,1,2,3,4
        print(i, end=" ")
    print()

    # range的其他用法
    # Java: for (int i = 1; i <= 5; i++)
    print("range(1, 6):")
    for i in range(1, 6):  # 从1到5
        print(i, end=" ")
    print()

    # Java: for (int i = 0; i < 10; i += 2)
    print("range(0, 10, 2):")
    for i in range(0, 10, 2):  # 步长为2
        print(i, end=" ")
    print()

    # for循环 - 遍历列表
    # Java: for (int num : numbers)
    numbers = [1, 2, 3, 4, 5]
    print("\n遍历列表:")
    for num in numbers:
        print(num, end=" ")
    print()

    # 带索引的遍历
    # Java: for (int i = 0; i < numbers.size(); i++)
    print("带索引遍历:")
    for i, num in enumerate(numbers):
        print(f"索引{i}: {num}")

    # 遍历字典
    # Java: for (Map.Entry<String, Integer> entry : map.entrySet())
    ages = {"Alice": 25, "Bob": 30}
    print("遍历字典:")
    for name, age in ages.items():
        print(f"{name}: {age}")

    # while循环
    # Java: while (i < 5)
    print("\nwhile循环:")
    i = 0
    while i < 5:
        print(i, end=" ")
        i += 1  # Python没有i++，用i += 1
    print()

    # break和continue
    # Java: break和continue用法相同
    print("\nbreak和continue:")
    for i in range(10):
        if i == 5:
            break  # 跳出循环
        if i % 2 == 0:
            continue  # 跳过本次循环
        print(i, end=" ")
    print()

    # Python特有：for-else和while-else
    # 如果循环正常结束（没有break），执行else
    print("\nfor-else:")
    for i in range(5):
        print(i, end=" ")
    else:
        print("循环正常结束")


# ============================================================
# 9. 函数
# Functions
# ============================================================

def functions_demo():
    """
    函数定义和使用
    
    Java对应：
    public void functionsDemo() {
        // 定义函数
        public int add(int a, int b) {
            return a + b;
        }
        
        // 调用函数
        int result = add(3, 5);
        
        // 重载函数（Java支持，Python不支持）
        public int add(int a, int b, int c) {
            return a + b + c;
        }
        
        // 可变参数
        public int sum(int... numbers) {
            int total = 0;
            for (int num : numbers) {
                total += num;
            }
            return total;
        }
    }
    """
    print("\n" + "=" * 50)
    print("9. 函数")
    print("=" * 50)

    # 定义简单函数
    # Java: public int add(int a, int b) { return a + b; }
    def add(a, b):
        """函数文档字符串"""
        return a + b

    result = add(3, 5)
    print(f"add(3, 5) = {result}")

    # 默认参数
    # Java: 需要方法重载实现
    def greet(name, greeting="Hello"):
        return f"{greeting}, {name}!"

    print(greet("Alice"))  # 使用默认值
    print(greet("Bob", "Hi"))  # 指定值

    # 关键字参数
    # Java: 不支持，需要按顺序传参
    print(greet(greeting="Hey", name="Charlie"))

    # 可变参数
    # Java: public int sum(int... numbers)
    def sum_all(*args):  # *args接收任意数量的位置参数
        return sum(args)

    print(f"sum_all(1, 2, 3, 4, 5) = {sum_all(1, 2, 3, 4, 5)}")

    # 可变关键字参数
    # Java: 不支持
    def print_info(**kwargs):  # **kwargs接收任意数量的关键字参数
        for key, value in kwargs.items():
            print(f"  {key}: {value}")

    print("print_info:")
    print_info(name="Alice", age=25, city="Beijing")

    # 返回多个值（Python特有）
    # Java: 需要创建对象或数组
    def get_stats(numbers):
        return min(numbers), max(numbers), sum(numbers) / len(numbers)


# ============================================================
# 10. 类和对象
# Classes and Objects
# ============================================================

def classes_demo():
    """
    类和对象

    Java对应：
    public class Person {
        // 成员变量
        private String name;
        private int age;

        // 构造方法
        public Person(String name, int age) {
            this.name = name;
            this.age = age;
        }

        // 方法
        public void sayHello() {
            System.out.println("Hello, I'm " + name);
        }

        // Getter/Setter
        public String getName() {
            return name;
        }

        public void setName(String name) {
            this.name = name;
        }

        public int getAge() {
            return age;
        }

        public void setAge(int age) {
            this.age = age;
        }

        // 继承
        public class Student extends Person {
            private String studentId;

            public Student(String name, int age, String studentId) {
                super(name, age);
                this.studentId = studentId;
            }
        }
    }
    """
    print("\n" + "=" * 50)
    print("10. 类和对象")
    print("=" * 50)

    # 定义类
    # Python中所有类默认继承自object
    class Person:
        # 类变量（类似Java的静态变量）
        species = "Human"

        # 构造方法（对应Java的构造器）
        # Java: public Person(String name, int age)
        def __init__(self, name, age):
            # 实例变量
            self.name = name
            self.age = age

        # 实例方法（对应Java的成员方法）
        # Java: public void sayHello()
        def say_hello(self):
            print(f"Hello, I'm {self.name} and I'm {self.age} years old")

        # 特殊方法：字符串表示
        # Java: public String toString()
        def __str__(self):
            return f"Person(name={self.name}, age={self.age})"

        # Getter方法（Python通常直接访问属性，但可以定义property）
        @property
        def name_property(self):
            return self.name

        # Setter方法
        @name_property.setter
        def name_property(self, name):
            self.name = name

        # 类方法（类似Java的静态方法）
        @classmethod
        def get_species(cls):
            return cls.species

    # 创建对象
    # Java: Person person = new Person("Alice", 25);
    person = Person("Alice", 25)

    # 访问属性
    # Java: person.getName()
    print(f"Name: {person.name}")
    print(f"Age: {person.age}")
    print(f"Species: {Person.species}")

    # 调用方法
    # Java: person.sayHello()
    person.say_hello()

    # 使用字符串表示
    # Java: person.toString()
    print(f"String representation: {person}")

    # 修改属性
    # Java: person.setName("Alicia")
    person.name = "Alicia"
    person.say_hello()

    # 使用property
    print(f"Name via property: {person.name_property}")
    person.name_property = "Alice"
    print(f"After setting via property: {person.name_property}")

    # 继承示例
    class Student(Person):
        # Java: public Student(String name, int age, String studentId)
        def __init__(self, name, age, student_id):
            # 调用父类构造方法
            # Java: super(name, age)
            super().__init__(name, age)
            self.student_id = student_id

        # 重写方法
        def say_hello(self):
            # Java: super.sayHello()
            super().say_hello()
            print(f"My student ID is {self.student_id}")

        def __str__(self):
            return f"Student(name={self.name}, age={self.age}, student_id={self.student_id})"

    # 创建子类对象
    student = Student("Bob", 20, "S12345")
    student.say_hello()
    print(f"Student: {student}")


# ============================================================
# 11. 异常处理
# Exception Handling
# ============================================================

def exception_handling():
    """
    异常处理

    Java对应：
    public void exceptionHandling() {
        // try-catch
        try {
            int result = 10 / 0;
        } catch (ArithmeticException e) {
            System.out.println("Cannot divide by zero: " + e.getMessage());
        } catch (Exception e) {
            System.out.println("General exception: " + e.getMessage());
        } finally {
            System.out.println("This always executes");
        }

        // 抛出异常
        public void validateAge(int age) throws IllegalArgumentException {
            if (age < 0) {
                throw new IllegalArgumentException("Age cannot be negative");
            }
        }

        // 自定义异常
        class CustomException extends Exception {
            public CustomException(String message) {
                super(message);
            }
        }
    }
    """
    print("\n" + "=" * 50)
    print("11. 异常处理")
    print("=" * 50)

    # try-except (对应Java的try-catch)
    try:
        # Java: int result = 10 / 0;
        result = 10 / 0
    except ZeroDivisionError as e:
        # Java: catch (ArithmeticException e)
        print(f"Cannot divide by zero: {e}")
    except Exception as e:
        # Java: catch (Exception e)
        print(f"General exception: {e}")
    else:
        # Python特有：如果没有异常发生
        print("No exception occurred")
    finally:
        # Java: finally
        print("This always executes")

    # 抛出异常
    # Java: throw new IllegalArgumentException("Invalid value")
    def validate_age(age):
        if age < 0:
            raise ValueError("Age cannot be negative")
        return age

    try:
        validate_age(-5)
    except ValueError as e:
        print(f"Validation error: {e}")

    # 自定义异常
    # Java: class CustomException extends Exception
    class CustomException(Exception):
        def __init__(self, message):
            super().__init__(message)

    try:
        raise CustomException("This is a custom exception")
    except CustomException as e:
        print(f"Caught custom exception: {e}")


# ============================================================
# 12. 模块和导入
# Modules and Imports
# ============================================================

def modules_demo():
    """
    模块和导入

    Java对应：
    // Java使用package和import
    package com.example.utils;

    import java.util.List;
    import java.util.ArrayList;

    public class Calculator {
        public static int add(int a, int b) {
            return a + b;
        }
    }

    // 在其他文件中使用
    import com.example.utils.Calculator;

    public class Main {
        public static void main(String[] args) {
            int result = Calculator.add(3, 5);
        }
    }
    """
    print("\n" + "=" * 50)
    print("12. 模块和导入")
    print("=" * 50)

    # Python导入方式
    # 假设有模块文件: mymodule.py

    # Java: import java.util.List;
    import math  # 导入整个模块
    print(f"math.sqrt(16) = {math.sqrt(16)}")

    # Java: import java.util.ArrayList;
    from math import pi, sin  # 导入特定函数/变量
    print(f"Pi value: {pi}")
    print(f"sin(pi/2) = {sin(pi / 2)}")

    # Java: import java.util.*;
    # from math import *  # 导入所有（不推荐）

    # 别名导入
    # Java: import java.util.Date as JDate;
    import datetime as dt
    print(f"Current date: {dt.datetime.now().date()}")

    # 模块搜索路径
    import sys
    print("\nModule search paths:")
    for path in sys.path[:3]:  # 只显示前3个
        print(f"  {path}")

    # 创建和使用自定义模块的示例
    # 假设有文件 mymodule.py，内容如下：
    """
    # mymodule.py
    def hello(name):
        return f"Hello, {name}!"

    class Calculator:
        @staticmethod
        def add(a, b):
            return a + b
    """
    # 使用方式：
    # import mymodule
    # result = mymodule.hello("Alice")
    # calc = mymodule.Calculator()


# ============================================================
# 13. 文件操作
# File Operations
# ============================================================

def file_operations():
    """
    文件操作

    Java对应：
    public void fileOperations() {
        // 读取文件
        try (BufferedReader reader = new BufferedReader(new FileReader("file.txt"))) {
            String line;
            while ((line = reader.readLine()) != null) {
                System.out.println(line);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 写入文件
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("output.txt"))) {
            writer.write("Hello, World!");
            writer.newLine();
            writer.write("This is a test.");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    """
    print("\n" + "=" * 50)
    print("13. 文件操作")
    print("=" * 50)

    # 写入文件
    # Java: FileWriter + BufferedWriter
    print("Writing to file...")
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write("Hello, World!\n")
        f.write("This is a test file.\n")
        f.write("Python file operations are simple.\n")

    # 读取文件
    # Java: FileReader + BufferedReader
    print("\nReading entire file:")
    with open("test.txt", "r", encoding="utf-8") as f:
        content = f.read()
        print(content)

    print("\nReading line by line:")
    with open("test.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            print(f"Line {i}: {line.strip()}")

    # 追加文件
    # Java: FileWriter(file, true)
    print("\nAppending to file...")
    with open("test.txt", "a", encoding="utf-8") as f:
        f.write("Appended line.\n")

    # 读取所有行到列表
    # Java: Files.readAllLines()
    with open("test.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        print(f"\nTotal lines: {len(lines)}")

    # 使用with语句自动关闭文件（推荐）
    # 对应Java的try-with-resources

    # 清理测试文件
    import os
    if os.path.exists("test.txt"):
        os.remove("test.txt")
        print("Test file deleted")


# ============================================================
# 14. 列表推导式和生成器
# List Comprehensions and Generators
# ============================================================

def comprehensions_and_generators():
    """
    列表推导式和生成器（Python特有特性）

    Java对应：
    // Java 8+ Stream API类似
    public void streamOperations() {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9, 10);

        // 过滤和映射
        List<Integer> squares = numbers.stream()
            .filter(n -> n % 2 == 0)
            .map(n -> n * n)
            .collect(Collectors.toList());

        // 生成范围
        List<Integer> range = IntStream.range(1, 11)
            .boxed()
            .collect(Collectors.toList());
    }
    """
    print("\n" + "=" * 50)
    print("14. 列表推导式和生成器")
    print("=" * 50)

    # 列表推导式（List Comprehension）
    # Java: numbers.stream().filter(n -> n % 2 == 0).collect(Collectors.toList())
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # 过滤偶数
    even_numbers = [n for n in numbers if n % 2 == 0]
    print(f"Even numbers: {even_numbers}")

    # 计算平方
    squares = [n * n for n in numbers]
    print(f"Squares: {squares}")

    # 过滤并计算平方
    even_squares = [n * n for n in numbers if n % 2 == 0]
    print(f"Even squares: {even_squares}")

    # 字典推导式
    square_dict = {n: n * n for n in numbers}
    print(f"Square dictionary: {square_dict}")

    # 集合推导式
    unique_squares = {n * n for n in numbers}
    print(f"Unique squares: {unique_squares}")

    # 生成器表达式（惰性求值）
    # Java: Stream API
    gen = (n * n for n in numbers if n % 2 == 0)
    print("Generator values:", end=" ")
    for value in gen:
        print(value, end=" ")
    print()

    # 生成器函数
    # Java: 需要实现Iterator
    def fibonacci(limit):
        a, b = 0, 1
        while a < limit:
            yield a  # 使用yield而不是return
            a, b = b, a + b

    print("Fibonacci numbers (up to 20):")
    for num in fibonacci(20):
        print(num, end=" ")
    print()


# ============================================================
# 15. 装饰器
# Decorators (Python特有)
# ============================================================

def decorators_demo():
    """
    装饰器（Python高级特性）

    Java对应：
    // Java使用注解（Annotations）和AOP实现类似功能
    @Retention(RetentionPolicy.RUNTIME)
    @Target(ElementType.METHOD)
    public @interface LogExecutionTime {
    }

    // AOP切面
    @Aspect
    @Component
    public class LoggingAspect {
        @Around("@annotation(LogExecutionTime)")
        public Object logExecutionTime(ProceedingJoinPoint joinPoint) throws Throwable {
            long start = System.currentTimeMillis();
            Object result = joinPoint.proceed();
            long end = System.currentTimeMillis();
            System.out.println(joinPoint.getSignature() + " executed in " + (end - start) + "ms");
            return result;
        }
    }
    """
    print("\n" + "=" * 50)
    print("15. 装饰器")
    print("=" * 50)

    # 简单装饰器
    def simple_decorator(func):
        def wrapper():
            print("Before function call")
            func()
            print("After function call")

        return wrapper

    @simple_decorator
    def say_hello():
        print("Hello!")

    print("Simple decorator:")
    say_hello()

    # 带参数的装饰器
    def repeat(times):
        def decorator(func):
            def wrapper(*args, **kwargs):
                for i in range(times):
                    print(f"Call {i + 1}:")
                    result = func(*args, **kwargs)
                return result

            return wrapper

        return decorator

    @repeat(3)
    def greet(name):
        print(f"Hello, {name}!")

    print("\nParameterized decorator:")
    greet("Alice")

    # 实用的装饰器：计时器
    import time

    def timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            print(f"{func.__name__} executed in {end - start:.6f} seconds")
            return result

        return wrapper

    @timer
    def slow_function():
        time.sleep(0.1)
        return "Done"

    print("\nTimer decorator:")
    slow_function()

    # 使用functools.wraps保持函数元信息
    from functools import wraps

    def logged(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            return func(*args, **kwargs)

        return wrapper

    @logged
    def add(a, b):
        """Add two numbers"""
        return a + b

    print("\nWith functools.wraps:")
    print(f"Function name: {add.__name__}")
    print(f"Function doc: {add.__doc__}")
    result = add(3, 5)
    print(f"Result: {result}")


# ============================================================
# 主函数
# Main Function
# ============================================================
def main():
    """
    主函数 - 执行所有演示

    Java对应：
    public static void main(String[] args) {
        PythonForJavaDemo demo = new PythonForJavaDemo();
        demo.variablesAndTypes();
        demo.stringOperations();
        // ... 其他方法调用
    }
    """
    print("Python基础语法 - 面向Java程序员")
    print("=" * 60)

    # 执行所有演示函数
    functions = [
        variables_and_types,
        string_operations,
        list_operations,
        dictionary_operations,
        set_operations,
        tuple_operations,
        control_flow_if,
        loops,
        functions_demo,
        classes_demo,
        exception_handling,
        modules_demo,
        file_operations,
        comprehensions_and_generators,
        decorators_demo,
    ]

    for i, func in enumerate(functions, 1):
        try:
            func()
        except Exception as e:
            print(f"\nError in {func.__name__}: {e}")
            import traceback
            traceback.print_exc()

        # 在函数之间添加分隔线，但不要暂停
        if i < len(functions):
            print("\n" + "=" * 60)
            # 移除 input("按Enter键继续...")
            # 添加一个短暂延时，让输出更清晰（可选）
            import time
            time.sleep(0.5)  # 0.5秒间隔
            print("=" * 60)

    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)

    # 额外提示
    print("\n额外提示：")
    print("1. Python使用缩进表示代码块（通常4个空格）")
    print("2. Python是动态类型语言，不需要声明变量类型")
    print("3. Python有垃圾回收，不需要手动管理内存")
    print("4. Python使用'#'表示注释，而不是'//'")
    print("5. Python没有分号结束语句（除非一行有多条语句）")
    print("6. Python使用下划线命名法（snake_case），而不是驼峰命名法（camelCase）")


# ============================================================
# Python标准入口点
# ============================================================

if __name__ == "__main__":
    """
    Python的入口点约定

    Java对应：
    public static void main(String[] args) {
        // 程序入口
    }

    解释：
    - __name__ 是Python的特殊变量
    - 当文件直接运行时，__name__ 等于 "__main__"
    - 当文件被导入时，__name__ 等于模块名
    - 这样可以区分是直接运行还是被导入
    """
    main()

    # 其他常见用法
    # if __name__ == "__main__":
    #     import sys
    #     if len(sys.argv) > 1:
    #         print(f"命令行参数: {sys.argv[1:]}")
    #     main()

# ============================================================
# 额外补充：Python与Java的主要区别总结
# ============================================================

"""
Python与Java主要区别总结：

1. 语法风格
   - Python：使用缩进，简洁，动态类型
   - Java：使用大括号，严谨，静态类型

2. 类型系统
   - Python：动态类型，运行时类型检查
   - Java：静态类型，编译时类型检查

3. 性能
   - Python：解释执行，一般较慢
   - Java：编译为字节码，JIT优化，性能较好

4. 跨平台
   - Python：需要解释器
   - Java：一次编译，到处运行（JVM）

5. 内存管理
   - Python：自动垃圾回收（引用计数+循环检测）
   - Java：自动垃圾回收（分代收集）

6. 并发模型
   - Python：GIL限制，多线程不是真并行
   - Java：真正的多线程并行

7. 生态系统
   - Python：数据科学、机器学习、Web开发（Django/Flask）
   - Java：企业应用、Android开发、大数据（Hadoop）

8. 学习曲线
   - Python：入门简单，语法简洁
   - Java：入门较难，概念较多

9. 开发速度
   - Python：开发快速，适合原型和脚本
   - Java：开发较慢，适合大型系统

10. 部署
    - Python：直接运行源代码或打包
    - Java：编译为jar/war部署

常用对应关系：
1. Java的ArrayList → Python的list
2. Java的HashMap → Python的dict
3. Java的HashSet → Python的set
4. Java的String → Python的str
5. Java的int/double → Python的int/float
6. Java的System.out.println → Python的print
7. Java的for循环 → Python的for-in循环
8. Java的if-else → Python的if-elif-else
9. Java的try-catch-finally → Python的try-except-else-finally
10. Java的class → Python的class
"""

# ============================================================
# 示例：快速参考卡片
# ============================================================

QUICK_REFERENCE = """
Python快速参考卡片（Java开发者版）

变量定义：
  Java: int x = 10;
  Python: x = 10

条件语句：
  Java: if (x > 0) { ... } else if (x < 0) { ... } else { ... }
  Python: if x > 0: ... elif x < 0: ... else: ...

循环：
  Java: for (int i = 0; i < 10; i++) { ... }
  Python: for i in range(10): ...

  Java: for (String s : list) { ... }
  Python: for s in list: ...

函数定义：
  Java: public int add(int a, int b) { return a + b; }
  Python: def add(a, b): return a + b

类定义：
  Java: class Person { private String name; public Person(String name) {...} }
  Python: class Person: def __init__(self, name): self.name = name

异常处理：
  Java: try {...} catch (Exception e) {...} finally {...}
  Python: try: ... except Exception as e: ... finally: ...

列表操作：
  Java: list.add(1); list.get(0); list.size();
  Python: list.append(1); list[0]; len(list)

字典操作：
  Java: map.put("key", "value"); map.get("key"); map.containsKey("key");
  Python: dict["key"] = "value"; dict["key"]; "key" in dict
"""

# 保存为完整文件
if __name__ == "__main__":
    # 可选：将快速参考保存到文件
    with open("python_for_java_quick_ref.txt", "w", encoding="utf-8") as f:
        f.write(QUICK_REFERENCE)
    print("\n快速参考已保存到: python_for_java_quick_ref.txt")
