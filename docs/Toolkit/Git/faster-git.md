# faster git



> 这里主要记录笔者学习 Datawhale [faster-git](https://github.com/datawhalechina/faster-git)教程的笔记
>
> 推荐Git练习网站 [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN)

## 第一章 Git简介

### 分布式版本控制系统

每次将代码和修改记录克隆下来。

中央服务器用来“交换”大家的修改记录。

### 查看&修改设置

```bash
git config --list
git config --global A "B"
```

## 第二章 Git基础命令

### 忽略文件

 .gitignore

### git init

创建 .git 子目录

### git add

追踪文件，暂存`修改状态`,将内容添加到下次提交中

### git commit

提交更新

弹出文件，第一行写提交说明，写完后关掉该文件

或者`git commit -m ""`

### git commit -a

跳过`git add`

### git clone

直接加链接，链接后也可添加自定义的名字

### git status

查看状态

`git status --short` 更简单，直观的状态

### git diff

修改后尚未暂存，尚未git add 的文件那些部分被修改

### git diff --staged

已经暂存，还未提交的内容，和上次提交的差异

### git rm

从git中移除文件



## 第三章 Git分支管理

每个人开发独立，互不影响

### git branch

查看所有分支

*标记当前分支

后面加分支名即可创建新的分支，但创建分支后并不改变 * 的位置

#### -d

删除分支

#### -D

强行删除分支

#### -m

重命名

重命名后的分支再push到远端，需要先push到远端，再删除远端的旧名字的分支

### git checkout

切换分支

### touch

创建新的文件

### git log --oneline

检查当前 Git 记录

### git merge

主分支 A

~ B

B 仍然存在，只是把 B 上的文件合并到主分支 A 上，如果 A ，B对文件的修改产生冲突

解决方案：

* 手动合并

* 放弃合并
* mergetool

### git remote -v

看远程的信息

### git push origin

推送到远程

后面加本地的分支

#### --delete

删除远程分支

### 长期分支

同时多个分支，定期把短期分支合并到develop分支，主分支只保留稳定版。

### 短期/主题分支

平时用来解决 issue，bug都改完了，再合并到其他分支。 

## 第四章 Git 工具

### git log

提交日志

commit 后跟的 40 位的字符，是 SHA-1 哈希值

#### --abbrev-commit

显示 7 位 SHA-1 的简写

#### --pretty=oneline

简化输出

### git show

后面加 SHA-1 值，查看具体某一次提交的信息

可以用前几位代替，只要不冲突多短都行，默认7位

#### stable

看指定分支最后一次提交信息

### git reflog

引用日志

近几月的 HEAD 和分支引用的历史

### git add -i

i is for interactive 

进入交互模式，修改大量文件，拆成多个提交

#### 2

暂存哪个文件

#### 3

取消暂存



## 第五章 Git 内部原理

`.git`隐藏目录

### objectes 对象储存

三个对象：

* 数据
* 树
* 提交

SHA-1 哈希值：校验和

40位

校验和前两位命名子目录，后38位命名文件名

子目录/文件名

#### git cat-file -t

查看类型

#### git cat-file -t

查看内容



### objects目录

省地，增效，多个对象打包成 “包文件” `.pack`

####  git gc

手动打包松散的对象

索引文件 `.idx`



###  refs目录—— 引用

引用类型

* heads
* remotes
* tags

#### git tag -a v1.0 <commitId>

基于commit打tag

#### HEAD 引用

* 分支级别
* 代码库级别

#### remotes 引用

远程仓库分支最后一次提交

#### tags 引用

发布版本管理

#### stash

暂存当前修改

### config文件 —— 引用规范

由`git remote add origin` 生成



### config文件 —— 环境变量

#### git config

1. 系统变量

   --system

2. 用户变量

   --global

3. 本地项目变量

​	--local



## 第六章 GitFlow工作流实战



## 第七章 Git提交规范

### Commit Message

```git
<header>
<BLANK LINE>
<body>
<BLANK LINE>
<footer>
```

`<header>`

```
<type>(<scope>): <short summary>
```

`<type>`

* build：构建
* ci：持续集成
* docs：文档
* feat：新功能
* fix：bug修复
* perf：性能
* refactor 重构
* test：测试

`<scope>`

改动的范围

`<summary>`

祈使句、现在时。

`<body>`

更详细的描述，祈使句、现在时。

`<footer>`

可选项，特殊改动

#### 自动化校验

Git Gooks功能

[官方文档](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks)

###  Author & Committer

Author 原始纂写该提交的作者

Committer 应用该提交的人



### Changed files

```bash
git diff # 看改动
git add 
git status 
git commit
```

2 个bug修复分成 2 次提交

经常提交，经常分享

关键分支一定要测试

不提交编译输出，日志，中间产物，用[.gitignore](https://github.com/github/gitignore)略。

不提交密码，删历史记录困难

配置文件用模板放本地

```bash
git reset <file> # 移除尚未被commit的文件
git clean -f # 移除未被追踪的中间文件
git checkout <file> # 回退改动
```



### Hash & Parent

```bash
git rebase -i <commit> # 合并到主分支之前对相关提交进行合并，废弃，修改信息……
git push -f #覆盖 禁止对主分支用
git pull --rebase #对本地提交进行更新
```



## 第八章 Github/Gitee使用说明

### 仓库介绍

#### Fork

创建仓库副本

#### Watch

邮箱接送仓库推送

#### Issues

议题，仓库内容，bug/feat

#### Pull Request

最主要工作单元

“我修改好了你的代码，现在请求你把代码拉回主仓库中”

#### Action

自动化构建

#### Projects

项目板

#### Wiki

介绍性内容

#### Security

安全

#### Insight

代码贡献

#### discussion

讨论区

#### 提交issue



### 提交issue

有一些模板

叙述bug的步骤和环境

[How to create a Minimal, Reproducible Example - Help Center - Stack Overflow](https://stackoverflow.com/help/minimal-reproducible-example)



### 提交PR

fork后修改希望合并到上游仓库

尽量关联Issue

draft PR 



### 探索Github

[GitHunt – Trending Github Repositories (kamranahmed.info)](https://kamranahmed.info/githunt/)

[键盘快捷方式 - GitHub 文档](https://docs.github.com/zh/get-started/accessibility/keyboard-shortcuts)



### 高级搜索

