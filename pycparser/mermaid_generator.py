from . import c_ast
import html


class MermaidGenerator(object):
    """ Uses the same visitor pattern as c_ast.NodeVisitor, but modified to
        return a value from each visit method, using string accumulation in
        generic_visit.
    """

    def __init__(self):
        # Statements start with indentation of self.indent_level spaces, using
        # the _make_indent method
        #
        self.indent_level = 0
        self.stmt_seq = 0
        self.call_stack = []
        self.decl_node_hold = False

    def _make_indent(self):
        return ' ' * self.indent_level

    def _make_seq(self, node):
        return 'node_' + node.__class__.__name__ + '_' + str(self.stmt_seq)

    def _make_node(self, node, content, surround='[]'):
        if self.decl_node_hold:
            return content
        else:
            return self._make_seq(node) + surround[0] + "\"" + html.escape(content.replace('\n', '')) + "\"" + surround[
                1] + '\n';

    def _push_call_stack(self, node):
        self.call_stack.append(node)

    def _pop_call_stack(self):
        return self.call_stack.pop()

    def visit(self, node):
        self.stmt_seq += 1
        method = 'visit_' + node.__class__.__name__
        return getattr(self, method, self.generic_visit)(node)

    def generic_visit(self, node):
        # ~ print('generic:', type(node))
        if node is None:
            return ''
        else:
            return ''.join(self.visit(c) for c_name, c in node.children())

    def visit_Constant(self, n):
        return n.value

    def visit_ID(self, n):
        return n.name

    def visit_Pragma(self, n):
        ret = '#pragma'
        if n.string:
            ret += ' ' + n.string
        return ret

    def visit_ArrayRef(self, n):
        arrref = self._parenthesize_unless_simple(n.name)
        return arrref + '[' + self.visit(n.subscript) + ']'

    def visit_StructRef(self, n):
        sref = self._parenthesize_unless_simple(n.name)
        return sref + n.type + self.visit(n.field)

    def visit_FuncCall(self, n):
        fref = self._parenthesize_unless_simple(n.name)
        s = fref + '(' + self.visit(n.args) + ')'
        s = self._make_node(n, s)
        return s

    def visit_UnaryOp(self, n):
        operand = self._parenthesize_unless_simple(n.expr)
        if n.op == 'p++':
            s = '%s++' % operand
        elif n.op == 'p--':
            s = '%s--' % operand
        elif n.op == 'sizeof':
            # Always parenthesize the argument of sizeof since it can be
            # a name.
            s = 'sizeof(%s)' % self.visit(n.expr)
        else:
            s = '%s%s' % (n.op, operand)
        s = self._make_node(n, s)
        return s

    def visit_BinaryOp(self, n):
        lval_str = self._parenthesize_if(n.left,
                                         lambda d: not self._is_simple_node(d))
        rval_str = self._parenthesize_if(n.right,
                                         lambda d: not self._is_simple_node(d))
        s = '%s %s %s' % (lval_str, n.op, rval_str)
        s = self._make_node(n, s)
        return s

    def visit_Assignment(self, n):
        rval_str = self._parenthesize_if(
                n.rvalue,
                lambda n: isinstance(n, c_ast.Assignment))
        s = '%s %s %s' % (self.visit(n.lvalue), n.op, rval_str)
        s = self._make_node(n, s)
        return s

    def visit_IdentifierType(self, n):
        return ' '.join(n.names)

    def _visit_expr(self, n):
        if isinstance(n, c_ast.InitList):
            return '{' + self.visit(n) + '}'
        elif isinstance(n, c_ast.ExprList):
            return '(' + self.visit(n) + ')'
        else:
            return self.visit(n)

    def visit_Decl(self, n, no_type=False, nested=False):
        # no_type is used when a Decl is part of a DeclList, where the type is
        # explicitly only for the first declaration in a list.
        #
        node_hold = self.decl_node_hold
        if nested:
            self.decl_node_hold = True
        if no_type:
            s = n.name
        else:
            s = self._generate_decl(n)
        if n.bitsize: s += ' : ' + self.visit(n.bitsize)
        if n.init:
            s += ' = ' + self._visit_expr(n.init)
        s = self._make_node(n, s)
        if not nested:
            self.decl_node_hold = node_hold
        return s

    def visit_DeclList(self, n):
        s = self.visit(n.decls[0])
        if len(n.decls) > 1:
            s += ', ' + ', '.join(self.visit_Decl(decl, no_type=True)
                                    for decl in n.decls[1:])
        return s

    def visit_Typedef(self, n):
        s = ''
        if n.storage: s += ' '.join(n.storage) + ' '
        s += self._generate_type(n.type)
        return s

    def visit_Cast(self, n):
        s = '(' + self._generate_type(n.to_type) + ')'
        return s + ' ' + self._parenthesize_unless_simple(n.expr)

    def visit_ExprList(self, n):
        visited_subexprs = []
        for expr in n.exprs:
            visited_subexprs.append(self._visit_expr(expr))
        return ', '.join(visited_subexprs)

    def visit_InitList(self, n):
        visited_subexprs = []
        for expr in n.exprs:
            visited_subexprs.append(self._visit_expr(expr))
        return ', '.join(visited_subexprs)

    def visit_Enum(self, n):
        s = 'enum'
        if n.name: s += ' ' + n.name
        if n.values:
            s += ' {'
            for i, enumerator in enumerate(n.values.enumerators):
                s += enumerator.name
                if enumerator.value:
                    s += ' = ' + self.visit(enumerator.value)
                if i != len(n.values.enumerators) - 1:
                    s += ', '
            s += '}'
        return s

    def visit_FuncDef(self, n):
        self.decl_node_hold = True
        decl = self.visit_Decl(n.decl)
        self.decl_node_hold = False
        decl = self._make_node(n, decl)
        self.indent_level = 0
        body = self.visit(n.body)
        if n.param_decls:
            knrdecls = ';\n'.join(self.visit(p) for p in n.param_decls)
            return decl + '\n' + knrdecls + ';\n' + body + '\n'
        else:
            return decl + '\n' + body + '\n'

    def visit_FileAST(self, n):
        s = ''
        for ext in n.ext:
            if isinstance(ext, c_ast.FuncDef):
                s += self.visit(ext)
            elif isinstance(ext, c_ast.Pragma):
                s += self.visit(ext) + '\n'
            else:
                pass
                #s += self.visit(ext) + ';\n' # here are all typedef's
        return s

    def visit_Compound(self, n):
        s = self._make_indent()  # + '{\n'
        self.indent_level += 2
        if n.block_items:
            s += ''.join(self._generate_stmt(stmt) for stmt in n.block_items)
        self.indent_level -= 2
        s += self._make_indent()  # + '}\n'
        return s

    def visit_EmptyStatement(self, n):
        return ''  # ';'

    def visit_ParamList(self, n):
        return ', '.join(self.visit(param) for param in n.params)

    def visit_Return(self, n):
        s = 'return'
        if n.expr: s += ' ' + self.visit(n.expr)
        s = s + ';'
        s = self._make_node(n, s)
        return s

    def visit_Break(self, n):
        return 'break;'

    def visit_Continue(self, n):
        return 'continue;'

    def visit_TernaryOp(self, n):
        s = self._visit_expr(n.cond) + ' ? '
        s += self._visit_expr(n.iftrue) + ' : '
        s += self._visit_expr(n.iffalse)
        return s

    def visit_If(self, n):
        s = 'if ('
        if n.cond: s += self.visit(n.cond)
        s += ')'  # \n'
        s = self._make_node(n, s)
        s += self._make_seq(n) + " -- true --> "
        s += self._generate_stmt(n.iftrue, add_indent=True)
        if n.iffalse:
            s += self._make_seq(n) + " -- false --> "
            s += self._make_indent() + 'else\n'
            s += self._generate_stmt(n.iffalse, add_indent=True)
        return s

    def visit_For(self, n):
        s = 'for ('
        if n.init: s += self.visit(n.init)
        s += ';'
        if n.cond: s += ' ' + self.visit(n.cond)
        s += ';'
        if n.next: s += ' ' + self.visit(n.next)
        s += ')'  # \n'
        s = self._make_node(n, s)
        s += self._generate_stmt(n.stmt, add_indent=True)
        # TODO: break for statements
        return s

    def visit_While(self, n):
        s = 'while ('
        if n.cond: s += self.visit(n.cond)
        s += ')\n'
        s = self._make_node(n, s)
        s += self._generate_stmt(n.stmt, add_indent=True)
        return s

    def visit_DoWhile(self, n):
        s = 'do\n'
        s = self._make_node(n, s)
        s += self._generate_stmt(n.stmt, add_indent=True)
        swhile = self._make_indent() + 'while ('
        if n.cond: swhile += self.visit(n.cond)
        swhile += ');'
        swhile = self._make_node(n, swhile)
        s += swhile
        return s

    def visit_Switch(self, n):
        s = 'switch (' + self.visit(n.cond) + ')\n'
        s = self._make_node(n, s)
        s += self._generate_stmt(n.stmt, add_indent=True)
        return s

    def visit_Case(self, n):
        s = 'case ' + self.visit(n.expr) + ':\n'
        s = self._make_node(n, s)
        for stmt in n.stmts:
            s += self._generate_stmt(stmt, add_indent=True)
        return s

    def visit_Default(self, n):
        s = 'default:\n'
        s = self._make_node(n, s)
        for stmt in n.stmts:
            s += self._generate_stmt(stmt, add_indent=True)
        return s

    def visit_Label(self, n):
        s = n.name + ':\n' + self._generate_stmt(n.stmt)
        s = self._make_node(n, s)
        return s

    def visit_Goto(self, n):
        s = 'goto ' + n.name + ';'
        s = self._make_node(n, s)
        return s

    def visit_EllipsisParam(self, n):
        return '...'

    def visit_Struct(self, n):
        return self._generate_struct_union(n, 'struct')

    def visit_Typename(self, n):
        return self._generate_type(n.type)

    def visit_Union(self, n):
        return self._generate_struct_union(n, 'union')

    def visit_NamedInitializer(self, n):
        s = ''
        for name in n.name:
            if isinstance(name, c_ast.ID):
                s += '.' + name.name
            elif isinstance(name, c_ast.Constant):
                s += '[' + name.value + ']'
        s += ' = ' + self._visit_expr(n.expr)
        return s

    def visit_FuncDecl(self, n):
        return self._generate_type(n)

    def visit_Atomic(self, n):
        return '_Atomic(%s)' % self._generate_type(n.type)

    def _generate_struct_union(self, n, name):
        """ Generates code for structs and unions. name should be either
            'struct' or union.
        """
        s = name + ' ' + (n.name or '')
        if n.decls:
            s += '\n'
            s += self._make_indent()
            self.indent_level += 2
            s += '{\n'
            for decl in n.decls:
                s += self._generate_stmt(decl)
            self.indent_level -= 2
            s += self._make_indent() + '}'
        return s

    def _generate_stmt(self, n, add_indent=False, cond=''):
        """ Generation from a statement node. This method exists as a wrapper
            for individual visit_* methods to handle different treatment of
            some statements in this context.
        """
        typ = type(n)
        if add_indent: self.indent_level += 2
        indent = self._make_indent()
        if add_indent: self.indent_level -= 2

        if typ in (
                c_ast.Decl, c_ast.Assignment, c_ast.Cast, c_ast.UnaryOp,
                c_ast.BinaryOp, c_ast.TernaryOp, c_ast.FuncCall, c_ast.ArrayRef,
                c_ast.StructRef, c_ast.Constant, c_ast.ID, c_ast.Typedef,
                c_ast.ExprList):
            # These can also appear in an expression context so no semicolon
            # is added to them automatically
            #
            return indent + self.visit(n)  # + ';\n'
        elif typ in (c_ast.Compound,):
            # No extra indentation required before the opening brace of a
            # compound - because it consists of multiple lines it has to
            # compute its own indentation.
            #
            return self.visit(n)
        else:
            return indent + self.visit(n)  # + '\n'

    def _generate_decl(self, n):
        """ Generation from a Decl node.
        """
        s = ''
        if n.funcspec: s = ' '.join(n.funcspec) + ' '
        if n.storage: s += ' '.join(n.storage) + ' '
        s += self._generate_type(n.type)
        return s

    def _generate_type(self, n, modifiers=[], no_nested_node=False):
        """ Recursive generation from a type node. n is the type node.
            modifiers collects the PtrDecl, ArrayDecl and FuncDecl modifiers
            encountered on the way down to a TypeDecl, to allow proper
            generation from it.
        """
        typ = type(n)
        # ~ print(n, modifiers)

        if typ == c_ast.TypeDecl:
            s = ''
            if n.quals: s += ' '.join(n.quals) + ' '
            s += self.visit(n.type)

            nstr = n.declname if n.declname else ''
            # Resolve modifiers.
            # Wrap in parens to distinguish pointer to array and pointer to
            # function syntax.
            #
            for i, modifier in enumerate(modifiers):
                if isinstance(modifier, c_ast.ArrayDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '[' + self.visit(modifier.dim) + ']'
                elif isinstance(modifier, c_ast.FuncDecl):
                    if (i != 0 and isinstance(modifiers[i - 1], c_ast.PtrDecl)):
                        nstr = '(' + nstr + ')'
                    nstr += '(' + self.visit(modifier.args) + ')'
                elif isinstance(modifier, c_ast.PtrDecl):
                    if modifier.quals:
                        nstr = '* %s %s' % (' '.join(modifier.quals), nstr)
                    else:
                        nstr = '*' + nstr
            if nstr: s += ' ' + nstr
            return s
        elif typ == c_ast.Decl:
            return self._generate_decl(n.type)
        elif typ == c_ast.Typename:
            return self._generate_type(n.type)
        elif typ == c_ast.IdentifierType:
            return ' '.join(n.names) + ' '
        elif typ in (c_ast.ArrayDecl, c_ast.PtrDecl, c_ast.FuncDecl):
            return self._generate_type(n.type, modifiers + [n])
        else:
            return self.visit(n)

    def _parenthesize_if(self, n, condition):
        """ Visits 'n' and returns its string representation, parenthesized
            if the condition function applied to the node returns True.
        """
        s = self._visit_expr(n)
        if condition(n):
            return '(' + s + ')'
        else:
            return s

    def _parenthesize_unless_simple(self, n):
        """ Common use case for _parenthesize_if
        """
        return self._parenthesize_if(n, lambda d: not self._is_simple_node(d))

    def _is_simple_node(self, n):
        """ Returns True for nodes that are "simple" - i.e. nodes that always
            have higher precedence than operators.
        """
        return isinstance(n, (c_ast.Constant, c_ast.ID, c_ast.ArrayRef,
                              c_ast.StructRef, c_ast.FuncCall))
