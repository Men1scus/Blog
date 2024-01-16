# Git



> 这里主要记录笔者学习 Datawhale [faster-git](https://github.com/datawhalechina/faster-git)教程的笔记

> 推荐Git练习网站 [Learn Git Branching](https://learngitbranching.js.org/?locale=zh_CN)

## Lecture 1 Git简介

### 分布式版本控制系统

每次将代码和修改记录克隆下来。

中央服务器用来“交换”大家的修改记录。

### 查看&修改设置

```bash
git config --list
git config --global A "B"
```

## Lecture 2 Git基础命令

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

