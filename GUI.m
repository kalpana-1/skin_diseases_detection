function varargout = GUI(varargin)
%Beginning of initialization the code
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @GUI_OpeningFcn, ...
                   'gui_OutputFcn',  @GUI_OutputFcn, ...
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
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data 
% varargin   command line arguments to gui 

function GUI_OpeningFcn(hObject, eventdata, handles, varargin)
%Choose the default command line output for TestInterface
handles.output = hObject;
guidata(hObject, handles);

function varargout = GUI_OutputFcn(hObject, eventdata, handles) 
%Get default command line output from handles structure
varargout{1} = handles.output;
%Executes on button press via pushbuttons
function pushbutton1_Callback(hObject, eventdata, handles)
clc;
set(handles.pushbutton2,'Enable','on');
set(handles.pushbutton3,'Enable','on');
set(handles.output_text,'String', '');
cla(handles.axes_seg);
[filename,pathname] = uigetfile('*.jpg','Select lesion image');
I = imread(fullfile(pathname, filename));
[m, n] = size(I);
if (m < 300) || (n < 300)
    set(handles.output_text,'String', 'Invalid Image');
    set(handles.pushbutton2,'Enable','off');
    set(handles.pushbutton3,'Enable','off');
end
handles.I = I;
%%display an image in a GUI figure. specify it with the axes function in advance of calling imshow()
axes(handles.axes_org);
imshow(I);
% Update handles structure
guidata(hObject, handles);

function pushbutton2_Callback(hObject, eventdata, handles)
clc;
[Img, ai, ci, col, dia] = GUI_Seg(handles.I);
handles.Img_Seg = Img;
handles.ai = ai;
handles.ci = ci;
handles.col = col;
handles.dia = dia;
disp(['Assymetry Index - ', num2str(ai)]);
disp(['Compactness Index - ', num2str(ci)]);
disp(['Colour - ', num2str(col)]);
disp(['Diameter - ', num2str(dia)]);
axes(handles.axes_seg);
imshow(Img);
guidata(hObject, handles);



function pushbutton3_Callback(hObject, eventdata, handles)

[cat] = predict_net(handles.ai, handles.ci, handles.col, handles.dia);
set(handles.output_text,'String', cat);

function pushbutton4_Callback(hObject, eventdata, handles)

clc;
close (GUI);
[main] = GUIMain();
