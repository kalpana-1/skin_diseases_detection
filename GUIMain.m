function varargout = GUIMain(varargin)

gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUIMain_OpeningFcn, ...
                   'gui_OutputFcn',  @GUIMain_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
%Executes just before GUI is made visible
function GUIMain_OpeningFcn(hObject, eventdata, handles, varargin)
%Choose the default command line output for TestInterface
handles.output = hObject;
guidata(hObject, handles);

function varargout = GUIMain_OutputFcn(hObject, eventdata, handles) 
%Get default command line output from handles structure
varargout{1} = handles.output;
%Executes on button press via pushbuttons
function pushbutton1_Callback(hObject, eventdata, handles)

close(GUIMain);
[user] = GUI();



% --- Executes on button press in pushbutton4.
function pushbutton4_Callback(hObject, eventdata, handles)
% hObject    handle to pushbutton4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
